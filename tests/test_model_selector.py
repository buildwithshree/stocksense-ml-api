import pandas as pd
import numpy as np
import pytest
from app.models.selector import select_model_name, train_and_evaluate
from app.features.engineer import engineer_features, get_feature_columns


def make_ohlcv(n=300):
    np.random.seed(7)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.01,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_select_model_thin_data():
    assert select_model_name(50) == "Ridge"


def test_select_model_medium_data():
    assert select_model_name(150) == "RandomForest"


def test_select_model_standard_data():
    assert select_model_name(300) == "XGBoost"


def test_train_evaluate_returns_result():
    df_raw  = make_ohlcv(300)
    df      = engineer_features(df_raw)
    cols    = get_feature_columns(df)
    result  = train_and_evaluate(df, cols, "TEST.NS", force_model="XGBoost")
    assert result.model_name == "XGBoost"
    assert result.rmse >= 0
    assert result.r2 <= 1.0
    assert len(result.feature_importances) > 0
    assert result.model_size_mb > 0


def test_train_evaluate_ridge():
    df_raw  = make_ohlcv(120)
    df      = engineer_features(df_raw)
    cols    = get_feature_columns(df)
    result  = train_and_evaluate(df, cols, "TEST.NS", force_model="Ridge")
    assert result.model_name == "Ridge"
    assert result.rmse >= 0


def test_time_series_split_no_leakage():
    # Train end must be before test start — enforced by TimeSeriesSplit
    df_raw = make_ohlcv(300)
    df     = engineer_features(df_raw)
    cols   = get_feature_columns(df)
    result = train_and_evaluate(df, cols, "TEST.NS", force_model="XGBoost")
    from datetime import datetime
    train_end  = datetime.strptime(result.train_end,  "%Y-%m-%d")
    test_start = datetime.strptime(result.test_start, "%Y-%m-%d")
    assert train_end < test_start, "Data leakage: train_end >= test_start"
