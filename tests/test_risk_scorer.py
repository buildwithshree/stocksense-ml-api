import pandas as pd
import numpy as np
import pytest
from app.models.risk_scorer import compute_risk_score


def make_ohlcv(n=300, vol_multiplier=1.0):
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5 * vol_multiplier)
    return pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.01,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_risk_score_in_range():
    df = make_ohlcv()
    score, label = compute_risk_score(df, predicted_close=101.0, last_close=100.0)
    assert 0 <= score <= 100


def test_risk_label_valid():
    df = make_ohlcv()
    _, label = compute_risk_score(df, predicted_close=101.0, last_close=100.0)
    assert label in ["Low", "Moderate", "High", "Very High"]


def test_higher_volatility_gives_higher_score():
    df_calm   = make_ohlcv(vol_multiplier=0.2)
    df_choppy = make_ohlcv(vol_multiplier=5.0)
    score_calm,   _ = compute_risk_score(df_calm,   101.0, 100.0)
    score_choppy, _ = compute_risk_score(df_choppy, 101.0, 100.0)
    assert score_choppy >= score_calm, "Higher volatility should yield higher risk score"


def test_large_predicted_move_increases_score():
    df = make_ohlcv()
    score_small, _ = compute_risk_score(df, predicted_close=100.5,  last_close=100.0)
    score_large, _ = compute_risk_score(df, predicted_close=115.0,  last_close=100.0)
    assert score_large >= score_small
