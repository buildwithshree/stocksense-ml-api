import io
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from app.config import settings
from app.models.selector import TrainResult

logger = logging.getLogger(__name__)

# NullPool because FastAPI workers are short-lived — no persistent pool needed
engine = create_engine(
    settings.database_url,
    poolclass=NullPool,
    connect_args={"sslmode": "require"},
    echo=False,
)


def get_cached_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    """
    Return cached OHLCV from stock_cache if data exists and is fresh (<24h old).
    Returns None if cache miss or stale.
    """
    stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.cache_stale_hours)
    query = text("""
        SELECT trade_date, open_price, high_price, low_price, close_price, volume
        FROM stock_cache
        WHERE ticker = :ticker
        ORDER BY trade_date ASC
    """)
    freshness_query = text("""
        SELECT MAX(fetched_at) FROM stock_cache WHERE ticker = :ticker
    """)
    try:
        with engine.connect() as conn:
            last_fetch = conn.execute(freshness_query, {"ticker": ticker}).scalar()
            if last_fetch is None:
                return None
            # Make timezone-aware for comparison
            if last_fetch.tzinfo is None:
                last_fetch = last_fetch.replace(tzinfo=timezone.utc)
            if last_fetch < stale_cutoff:
                logger.info("Cache stale for %s, will re-fetch", ticker)
                return None
            df = pd.read_sql(query, conn, params={"ticker": ticker}, parse_dates=["trade_date"])
            df = df.rename(columns={
                "open_price": "Open", "high_price": "High",
                "low_price": "Low", "close_price": "Close", "volume": "Volume"
            })
            df = df.set_index("trade_date")
            logger.info("Cache HIT for %s — %d rows", ticker, len(df))
            return df
    except Exception as e:
        logger.error("Cache read error for %s: %s", ticker, e)
        return None


def write_ohlcv_cache(ticker: str, df: pd.DataFrame, currency: str = "USD") -> None:
    """
    Upsert OHLCV rows into stock_cache. Uses ON CONFLICT to avoid duplicates.
    """
    if df is None or df.empty:
        return
    upsert_sql = text("""
        INSERT INTO stock_cache (ticker, trade_date, open_price, high_price, low_price, close_price, volume, currency, fetched_at)
        VALUES (:ticker, :trade_date, :open_price, :high_price, :low_price, :close_price, :volume, :currency, NOW())
        ON CONFLICT (ticker, trade_date) DO UPDATE
            SET open_price  = EXCLUDED.open_price,
                high_price  = EXCLUDED.high_price,
                low_price   = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                volume      = EXCLUDED.volume,
                fetched_at  = NOW()
    """)
    rows = []
    for date, row in df.iterrows():
        rows.append({
            "ticker": ticker,
            "trade_date": date.date() if hasattr(date, "date") else date,
            "open_price": float(row["Open"]),
            "high_price": float(row["High"]),
            "low_price":  float(row["Low"]),
            "close_price": float(row["Close"]),
            "volume": int(row["Volume"]),
            "currency": currency,
        })
    try:
        with engine.begin() as conn:
            conn.execute(upsert_sql, rows)
        logger.info("Cache WRITE for %s — %d rows", ticker, len(rows))
    except Exception as e:
        logger.error("Cache write error for %s: %s", ticker, e)


def write_model_metrics(metrics: dict) -> None:
    sql = text("""
        INSERT INTO model_metrics
            (model_name, model_version, ticker, rmse, mae, r2_score,
             training_time_ms, inference_time_ms, model_size_mb, feature_count,
             train_start_date, train_end_date, test_start_date, test_end_date)
        VALUES
            (:model_name, :model_version, :ticker, :rmse, :mae, :r2_score,
             :training_time_ms, :inference_time_ms, :model_size_mb, :feature_count,
             :train_start_date, :train_end_date, :test_start_date, :test_end_date)
    """)
    try:
        with engine.begin() as conn:
            conn.execute(sql, metrics)
    except Exception as e:
        logger.error("model_metrics write error: %s", e)


def write_backtest_result(result: dict) -> None:
    sql = text("""
        INSERT INTO backtest_results
            (ticker, model_name, model_version, average_error, direction_accuracy, max_error, test_days)
        VALUES
            (:ticker, :model_name, :model_version, :average_error, :direction_accuracy, :max_error, :test_days)
    """)
    try:
        with engine.begin() as conn:
            conn.execute(sql, result)
    except Exception as e:
        logger.error("backtest_results write error: %s", e)


def save_model_artifact(ticker: str, result: TrainResult) -> None:
    """
    Persist a trained model (scaler + model + metadata, as a single joblib blob)
    to Postgres. This is what survives a Render free-tier restart — the
    in-memory ModelCache does not. One row per ticker; a new training run
    overwrites the old artifact via ON CONFLICT.
    """
    buf = io.BytesIO()
    joblib.dump(result, buf)
    artifact_bytes = buf.getvalue()

    sql = text("""
        INSERT INTO model_artifacts
            (ticker, model_name, model_version, artifact, rmse, mae, r2_score,
             feature_cols, train_start_date, train_end_date, test_start_date, test_end_date, created_at)
        VALUES
            (:ticker, :model_name, :model_version, :artifact, :rmse, :mae, :r2_score,
             :feature_cols, :train_start_date, :train_end_date, :test_start_date, :test_end_date, NOW())
        ON CONFLICT (ticker) DO UPDATE
            SET model_name       = EXCLUDED.model_name,
                model_version    = EXCLUDED.model_version,
                artifact         = EXCLUDED.artifact,
                rmse             = EXCLUDED.rmse,
                mae              = EXCLUDED.mae,
                r2_score         = EXCLUDED.r2_score,
                feature_cols     = EXCLUDED.feature_cols,
                train_start_date = EXCLUDED.train_start_date,
                train_end_date   = EXCLUDED.train_end_date,
                test_start_date  = EXCLUDED.test_start_date,
                test_end_date    = EXCLUDED.test_end_date,
                created_at       = NOW()
    """)
    try:
        with engine.begin() as conn:
            conn.execute(sql, {
                "ticker": ticker,
                "model_name": result.model_name,
                "model_version": result.model_version,
                "artifact": artifact_bytes,
                "rmse": result.rmse,
                "mae": result.mae,
                "r2_score": result.r2,
                "feature_cols": result.feature_cols,
                "train_start_date": result.train_start,
                "train_end_date": result.train_end,
                "test_start_date": result.test_start,
                "test_end_date": result.test_end,
            })
        logger.info("model_artifacts WRITE for %s (%s, %.2f KB)",
                     ticker, result.model_name, len(artifact_bytes) / 1024)
    except Exception as e:
        logger.error("model_artifacts write error for %s: %s", ticker, e)


def load_model_artifact(ticker: str) -> Optional[TrainResult]:
    """
    Load a persisted model for `ticker` if one exists and is not stale.
    Uses the same staleness window as stock_cache (settings.cache_stale_hours)
    so a persisted model and its underlying data go stale together.
    Returns None on cache miss, staleness, or any read/deserialize error —
    callers should treat None as "train fresh", never raise on this path.
    """
    stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.cache_stale_hours)
    query = text("""
        SELECT artifact, created_at FROM model_artifacts WHERE ticker = :ticker
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(query, {"ticker": ticker}).fetchone()
            if row is None:
                return None
            artifact_bytes, created_at = row
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if created_at < stale_cutoff:
                logger.info("Persisted model STALE for %s (created %s)", ticker, created_at)
                return None
            result = joblib.load(io.BytesIO(bytes(artifact_bytes)))
            logger.info("Persisted model LOADED for %s from DB", ticker)
            return result
    except Exception as e:
        logger.error("model_artifacts read error for %s: %s", ticker, e)
        return None