import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from app.config import settings

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
