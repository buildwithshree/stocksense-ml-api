import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from app.config import settings

logger = logging.getLogger(__name__)

# LSTM import is lazy — only when actually selected, to keep startup fast
_LSTM_AVAILABLE = False



@dataclass
class TrainResult:
    model_name: str
    model_version: str
    scaler: StandardScaler
    model: object
    feature_cols: list[str]
    rmse: float
    mae: float
    r2: float
    training_time_ms: int
    model_size_mb: float
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    feature_importances: list[str]   # top features in importance order


def select_model_name(n_rows: int) -> str:
    """
    Dynamic model selection based on available data volume.
    Rule:
      < 100 rows  → Ridge (minimal data, regularised linear)
      100–199     → RandomForest
      200–499     → XGBoost
      ≥ 500       → LSTM if torch available, else XGBoost
    This enforces the audit document principle: bigger ≠ better.
    """
    if n_rows < 100:
        return "Ridge"
    elif n_rows < 200:
        return "RandomForest"
    elif n_rows < 500:
        return "XGBoost"
    else:
        return "LSTM" if _LSTM_AVAILABLE else "XGBoost"


def train_and_evaluate(
    df_features,       # engineered DataFrame with Target column
    feature_cols: list[str],
    ticker: str,
    force_model: Optional[str] = None,
) -> TrainResult:
    """
    Full training pipeline:
    1. Scale features
    2. TimeSeriesSplit (no data leakage)
    3. Train selected model
    4. Evaluate on held-out test fold
    5. Return TrainResult with all metrics
    """
    X = df_features[feature_cols].values
    y = df_features["Target"].values
    dates = df_features.index

    n_rows = len(X)
    model_name = force_model or select_model_name(n_rows)
    logger.info("Selected model %s for %s (%d rows)", model_name, ticker, n_rows)

    # ── TimeSeriesSplit — NEVER random split on time-series data ─────────────
    # Use 5 splits; evaluate on the last fold (most recent data = test)
    if len(X) < 6:
        train_size = max(1, len(X) - 1)
        train_idx = np.arange(train_size)
        test_idx = np.arange(train_size, len(X))
    else:
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]   # last fold = most recent

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_start = str(dates[train_idx[0]].date())
    train_end   = str(dates[train_idx[-1]].date())
    test_start  = str(dates[test_idx[0]].date())
    test_end    = str(dates[test_idx[-1]].date())

    # ── Scale ──────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    model, y_pred, importances = _train_model(
        model_name, X_train_s, y_train, X_test_s, y_test, feature_cols
    )
    training_ms = int((time.time() - t0) * 1000)

    # ── Evaluate ──────────────────────────────────────────────────────────
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    # ── Model size estimate ───────────────────────────────────────────────
    model_size_mb = _estimate_size_mb(model)

    logger.info(
        "%s trained in %dms | RMSE=%.4f MAE=%.4f R2=%.4f size=%.2fMB",
        model_name, training_ms, rmse, mae, r2, model_size_mb
    )

    return TrainResult(
        model_name=model_name,
        model_version=settings.model_version,
        scaler=scaler,
        model=model,
        feature_cols=feature_cols,
        rmse=rmse,
        mae=mae,
        r2=r2,
        training_time_ms=training_ms,
        model_size_mb=model_size_mb,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        feature_importances=importances,
    )


def _train_model(name, X_train, y_train, X_test, y_test, feature_cols):
    if name == "LinearRegression":
        m = LinearRegression()
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        imp  = _coef_importance(m.coef_, feature_cols)
        return m, pred, imp

    elif name == "Ridge":
        m = Ridge(alpha=1.0)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        imp  = _coef_importance(m.coef_, feature_cols)
        return m, pred, imp

    elif name == "RandomForest":
        m = RandomForestRegressor(
            n_estimators=200, max_depth=10,
            min_samples_leaf=5, n_jobs=-1, random_state=42
        )
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        imp  = _tree_importance(m.feature_importances_, feature_cols)
        return m, pred, imp

    elif name == "XGBoost":
        m = XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            verbosity=0, n_jobs=-1
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        pred = m.predict(X_test)
        imp  = _tree_importance(m.feature_importances_, feature_cols)
        return m, pred, imp

    elif name == "LSTM":
        return _train_lstm(X_train, y_train, X_test, y_test, feature_cols)

    else:
        raise ValueError(f"Unknown model: {name}")


def _train_lstm(X_train, y_train, X_test, y_test, feature_cols):
    import torch
    import torch.nn as nn

    SEQ_LEN = min(30, len(X_train) // 4)
    HIDDEN   = 64
    LAYERS   = 2
    EPOCHS   = 30
    LR       = 0.001

    if len(X_train) <= SEQ_LEN or len(X_test) <= SEQ_LEN:
        logger.warning("Not enough rows for LSTM sequences, falling back to XGBoost")
        return _train_model("XGBoost", X_train, y_train, X_test, y_test, feature_cols)

    def make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs), np.array(ys)

    X_tr_seq, y_tr_seq = make_sequences(X_train, y_train, SEQ_LEN)
    X_te_seq, y_te_seq = make_sequences(X_test,  y_test,  SEQ_LEN)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_t   = torch.FloatTensor(X_tr_seq).to(device)
    y_tr_t   = torch.FloatTensor(y_tr_seq).to(device)
    X_te_t   = torch.FloatTensor(X_te_seq).to(device)

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden, layers):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze()

    model   = LSTMModel(X_train.shape[1], HIDDEN, LAYERS).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_te = model(X_te_t).cpu().numpy()

    pad = len(X_test) - len(pred_te)
    if pad > 0:
        pred_te = np.concatenate([np.full(pad, pred_te[0]), pred_te])

    model.eval()
    X_sample = torch.FloatTensor(X_tr_seq[-1:]).to(device).requires_grad_(True)
    output   = model(X_sample)
    output.sum().backward()
    grads    = X_sample.grad.abs().mean(dim=1).squeeze().cpu().numpy()
    ranked   = sorted(zip(feature_cols, grads), key=lambda x: x[1], reverse=True)
    imp      = [name for name, _ in ranked[:10]]

    return model, pred_te, imp


def _coef_importance(coef, feature_cols):
    abs_coef = np.abs(coef)
    ranked = sorted(zip(feature_cols, abs_coef), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:10]]


def _tree_importance(importances, feature_cols):
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:10]]


def _estimate_size_mb(model) -> float:
    import pickle
    try:
        return round(len(pickle.dumps(model)) / 1_048_576, 3)
    except Exception:
        return 0.0
