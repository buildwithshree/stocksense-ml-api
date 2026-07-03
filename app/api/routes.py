import logging
import time
from datetime import datetime, timezone
from typing import Union

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Response

from app.api.schemas import (
    PredictionResponse,
    TrainingStatusResponse,
    BacktestResponse,
    HealthResponse,
)
from app.config import settings
from app.db.database import (
    write_model_metrics,
    write_backtest_result,
    save_model_artifact,
    load_model_artifact,
)
from app.features.engineer import engineer_features, get_feature_columns
from app.models.backtester import run_backtest
from app.models.risk_scorer import compute_risk_score
from app.models.selector import train_and_evaluate
from app.pipeline.data_fetcher import fetch_ohlcv, get_company_name
from app.utils.cache import model_cache, training_registry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="UP", version=settings.model_version)


@router.get("/predict/{ticker}", response_model=Union[PredictionResponse, TrainingStatusResponse])
def predict(ticker: str, background_tasks: BackgroundTasks, response: Response):
    """
    Main prediction endpoint called by Spring Boot.
    Flow:
      1. Fetch/cache OHLCV
      2. Engineer features
      3. Resolve a trained model — in-memory cache -> persisted DB artifact ->
         cold (no model anywhere): kick off background training and return
         202 immediately instead of blocking the request for minutes.
      4. Predict next-day close
      5. Compute confidence interval + direction probability
      6. Compute risk score
      7. Return canonical response
    """
    ticker = ticker.upper().strip()
    t_start = time.time()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    try:
        df_raw, currency = fetch_ohlcv(ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # ── 2. Feature engineering ────────────────────────────────────────────────
    try:
        df = engineer_features(df_raw)
    except Exception as e:
        logger.error("Feature engineering failed for %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail="Feature engineering failed")

    feature_cols = get_feature_columns(df)
    last_close   = float(df_raw["Close"].iloc[-1])

    # ── 3. Model — in-memory cache -> persisted DB -> cold background train ───
    result = model_cache.get(ticker)

    if result is None:
        result = load_model_artifact(ticker)
        if result is not None:
            model_cache.set(ticker, result)
            logger.info("Warmed in-memory cache for %s from persisted DB artifact", ticker)

    if result is None:
        # Nothing cached, nothing persisted — this ticker needs a fresh train.
        if training_registry.is_in_progress(ticker):
            response.status_code = 202
            return TrainingStatusResponse(
                ticker=ticker,
                status="training",
                message=f"Model for {ticker} is already training. Retry shortly.",
                check_again_in_seconds=15,
            )

        if training_registry.try_start(ticker):
            background_tasks.add_task(_train_and_persist, ticker, df, feature_cols)
            response.status_code = 202
            return TrainingStatusResponse(
                ticker=ticker,
                status="training",
                message=(
                    f"No cached model for {ticker} yet — training started in the "
                    f"background. This is a one-time cost per ticker; retry in ~20-40s."
                ),
                check_again_in_seconds=20,
            )

        # Rare race: another request claimed the training slot between the
        # is_in_progress check above and try_start — still correct to 202.
        response.status_code = 202
        return TrainingStatusResponse(
            ticker=ticker,
            status="training",
            message=f"Model for {ticker} is training. Retry shortly.",
            check_again_in_seconds=15,
        )

    # ── 4. Predict ───────────────────────────────────────────────────────────
    t_inf = time.time()
    last_features = df[feature_cols].iloc[-1:].values
    last_features_scaled = result.scaler.transform(last_features)

    try:
        # LSTM requires sequence reshape — handled inside predict wrapper
        predicted_close = float(_predict(result, last_features_scaled, df, feature_cols))
    except Exception as e:
        logger.error("Inference failed for %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail="Prediction inference failed")

    inference_ms = int((time.time() - t_inf) * 1000)

    # ── 5. Confidence interval + direction probability ────────────────────────
    residual_std = result.rmse   # RMSE used as std proxy for interval
    confidence_lower = round(predicted_close - 1.96 * residual_std, 4)
    confidence_upper = round(predicted_close + 1.96 * residual_std, 4)

    expected_move = round((predicted_close - last_close) / last_close * 100, 4)

    # Direction probability: sigmoid-scaled from expected move
    direction_prob = round(float(1 / (1 + np.exp(-expected_move * 10))), 3)

    # ── 6. Risk score ─────────────────────────────────────────────────────────
    risk_score, risk_label = compute_risk_score(df_raw, predicted_close, last_close)

    # ── 7. Response ───────────────────────────────────────────────────────────
    total_ms = int((time.time() - t_start) * 1000)
    company_name = get_company_name(ticker)

    return PredictionResponse(
        ticker=ticker,
        company_name=company_name,
        currency=currency,
        last_close=round(last_close, 4),
        predicted_close=round(predicted_close, 4),
        expected_move_percent=expected_move,
        confidence_lower=confidence_lower,
        confidence_upper=confidence_upper,
        direction_probability=direction_prob,
        risk_score=risk_score,
        risk_label=risk_label,
        model_name=result.model_name,
        model_version=result.model_version,
        rmse=round(result.rmse, 4),
        inference_time_ms=inference_ms,
        top_features=result.feature_importances[:5],
        generated_at=datetime.now(timezone.utc),
    )


@router.get("/backtest/{ticker}", response_model=BacktestResponse)
def backtest(ticker: str):
    ticker = ticker.upper().strip()
    try:
        df_raw, _ = fetch_ohlcv(ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        bt = run_backtest(df_raw, ticker)
    except Exception as e:
        logger.error("Backtest failed for %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail="Backtest failed")

    # Persist result
    try:
        write_backtest_result(bt)
    except Exception as e:
        logger.warning("backtest_results write failed: %s", e)

    return BacktestResponse(
        **bt,
        ran_at=datetime.now(timezone.utc),
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _train_and_persist(ticker: str, df, feature_cols: list[str]) -> None:
    """
    Runs on FastAPI's BackgroundTasks executor, after the 202 has already
    been sent to the client. Trains the model, warms the in-memory cache,
    persists to Postgres (so a Render restart doesn't wipe it), writes
    model_metrics, then always releases the training-in-progress claim —
    including on failure, so a bad ticker doesn't stay wedged in "training"
    forever.
    """
    try:
        logger.info("Background training started for %s", ticker)
        result = train_and_evaluate(df, feature_cols, ticker)
        model_cache.set(ticker, result)

        try:
            save_model_artifact(ticker, result)
        except Exception as e:
            logger.warning("model_artifacts persist failed for %s: %s", ticker, e)

        try:
            write_model_metrics({
                "model_name":        result.model_name,
                "model_version":     result.model_version,
                "ticker":            ticker,
                "rmse":              result.rmse,
                "mae":               result.mae,
                "r2_score":          result.r2,
                "training_time_ms":  result.training_time_ms,
                "inference_time_ms": 0,
                "model_size_mb":     result.model_size_mb,
                "feature_count":     len(feature_cols),
                "train_start_date":  result.train_start,
                "train_end_date":    result.train_end,
                "test_start_date":   result.test_start,
                "test_end_date":     result.test_end,
            })
        except Exception as e:
            logger.warning("model_metrics write failed for %s: %s", ticker, e)

        logger.info("Background training complete for %s (%s)", ticker, result.model_name)
    except Exception as e:
        logger.error("Background training failed for %s: %s", ticker, e)
    finally:
        training_registry.finish(ticker)


def _predict(result, X_scaled, df, feature_cols) -> float:
    """Unified predict call — handles LSTM sequence reshaping if needed."""
    model = result.model

    if result.model_name == "LSTM":
        try:
            import torch
        except ImportError:
            raise ValueError("PyTorch not available for LSTM inference")

        SEQ_LEN = min(30, len(df) // 4)
        if len(df) < SEQ_LEN:
            raise ValueError("Not enough data for LSTM prediction sequence")
        seq_features = df[feature_cols].iloc[-SEQ_LEN:].values
        seq_scaled   = result.scaler.transform(seq_features)
        tensor       = torch.FloatTensor(seq_scaled).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            return float(model(tensor).item())
    else:
        return float(model.predict(X_scaled)[0])