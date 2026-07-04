import logging
import time

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from app.features.engineer import engineer_features, get_feature_columns
from app.models.selector import train_and_evaluate
from app.config import settings

logger = logging.getLogger(__name__)


def run_backtest(df_raw, ticker: str) -> dict:
    """
    Walk-forward backtest:
    For each day in the test window:
      - Use all prior data to train
      - Predict next day RETURN (not absolute price — model trains on
        Target_return, see selector.py / engineer.py for why)
      - Compare with actual return
    Returns aggregated metrics. average_error / max_error are in
    return-space (e.g. 0.02 = 2 percentage points of error), consistent
    with rmse/mae reported elsewhere in the system.
    """
    df = engineer_features(df_raw)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y = df["Target_return"].values   # was "Target" — must match what the model trains on
    dates = df.index

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    _, test_idx = splits[-1]

    errors = []
    direction_correct = 0
    direction_total = 0

    t0 = time.time()
    for i in test_idx:
        if i < 50:
            continue

        result = train_and_evaluate(
            df.iloc[:i],
            feature_cols,
            ticker,
            force_model="XGBoost",   # Use XGBoost for consistent backtest baseline
        )

        X_next = result.scaler.transform(X[i:i+1])
        y_pred   = float(result.model.predict(X_next)[0])   # predicted return
        y_actual = float(y[i])                                # actual return

        error = abs(y_pred - y_actual)
        errors.append(error)

        # Direction is now trivial and correct: a return's own sign IS its
        # direction — no need to compare against a separate previous price.
        pred_dir   = y_pred > 0
        actual_dir = y_actual > 0
        if pred_dir == actual_dir:
            direction_correct += 1
        direction_total += 1

    elapsed_ms = int((time.time() - t0) * 1000)

    if not errors:
        raise ValueError("No backtest results generated")

    test_days = len(errors)
    avg_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    dir_acc   = round((direction_correct / max(direction_total, 1)) * 100, 2)

    logger.info(
        "Backtest %s: %d days, avg_err=%.4f (return-space), dir_acc=%.2f%%, elapsed=%dms",
        ticker, test_days, avg_error, dir_acc, elapsed_ms
    )

    return {
        "ticker": ticker,
        "model_name": "XGBoost",
        "model_version": settings.model_version,
        "average_error": round(avg_error, 6),
        "direction_accuracy": dir_acc,
        "max_error": round(max_error, 6),
        "test_days": test_days,
    }