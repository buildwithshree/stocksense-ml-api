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
      - Predict next day close
      - Compare with actual
    Returns aggregated metrics.
    """
    df = engineer_features(df_raw)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y = df["Target"].values
    dates = df.index

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    _, test_idx = splits[-1]

    errors = []
    direction_correct = 0
    prev_actual = None

    t0 = time.time()
    for i in test_idx:
        if i < 50:
            continue
        X_train_bt = X[:i]
        y_train_bt = y[:i]

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train_bt)
        X_te_s = sc.transform(X[i:i+1])

        result = train_and_evaluate(
            df.iloc[:i],
            feature_cols,
            ticker,
            force_model="XGBoost",   # Use XGBoost for consistent backtest baseline
        )

        pred = result.scaler.transform(X[i:i+1])
        from sklearn.linear_model import Ridge
        # Use already-trained model from result
        y_pred = result.model.predict(pred)[0]
        y_actual = y[i]

        error = abs(y_pred - y_actual)
        errors.append(error)

        if prev_actual is not None:
            pred_dir   = y_pred > prev_actual
            actual_dir = y_actual > prev_actual
            if pred_dir == actual_dir:
                direction_correct += 1

        prev_actual = y_actual

    elapsed_ms = int((time.time() - t0) * 1000)

    if not errors:
        raise ValueError("No backtest results generated")

    test_days = len(errors)
    avg_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    dir_acc   = round((direction_correct / max(test_days - 1, 1)) * 100, 2)

    logger.info(
        "Backtest %s: %d days, avg_err=%.4f, dir_acc=%.2f%%",
        ticker, test_days, avg_error, dir_acc
    )

    return {
        "ticker": ticker,
        "model_name": "XGBoost",
        "model_version": settings.model_version,
        "average_error": round(avg_error, 4),
        "direction_accuracy": dir_acc,
        "max_error": round(max_error, 4),
        "test_days": test_days,
    }
