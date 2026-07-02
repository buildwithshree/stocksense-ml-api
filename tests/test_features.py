import pandas as pd
import numpy as np
import pytest
from app.features.engineer import engineer_features, get_feature_columns


def make_ohlcv(n=300):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.01,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_engineer_features_returns_dataframe():
    df = engineer_features(make_ohlcv())
    assert isinstance(df, pd.DataFrame)


def test_engineer_features_has_target():
    df = engineer_features(make_ohlcv())
    assert "Target" in df.columns


def test_engineer_features_no_nulls():
    df = engineer_features(make_ohlcv())
    assert df.isnull().sum().sum() == 0, "Engineered features contain NaN"


def test_feature_columns_excludes_ohlcv_and_target():
    df = engineer_features(make_ohlcv())
    cols = get_feature_columns(df)
    for excluded in ["Open", "High", "Low", "Close", "Volume", "Target"]:
        assert excluded not in cols


def test_rsi_bounded():
    df = engineer_features(make_ohlcv())
    assert df["RSI_14"].between(0, 100).all(), "RSI_14 out of 0-100 range"


def test_bb_pct_reasonable():
    df = engineer_features(make_ohlcv())
    # Most values should be between -0.5 and 1.5 (extreme moves can exceed)
    assert df["BB_pct"].between(-1, 2).sum() / len(df) > 0.95


def test_atr_positive():
    df = engineer_features(make_ohlcv())
    assert (df["ATR_14"] > 0).all(), "ATR_14 contains non-positive values"
