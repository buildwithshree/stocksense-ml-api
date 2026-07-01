import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all technical indicator features to OHLCV DataFrame.
    Input: DataFrame with columns Open, High, Low, Close, Volume
    Output: DataFrame with all features + target column 'Target' (next-day close)
    All features are computed on Close price unless stated otherwise.
    """
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Returns ──────────────────────────────────────────────────────────────
    df["return_1d"]  = close.pct_change(1)
    df["return_3d"]  = close.pct_change(3)
    df["return_5d"]  = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)

    # ── Moving Averages ───────────────────────────────────────────────────────
    for w in [5, 10, 20, 50, 200]:
        df[f"sma_{w}"] = close.rolling(w).mean()
        df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

    # Price relative to MAs (normalised)
    df["price_to_sma20"]  = close / df["sma_20"]
    df["price_to_sma50"]  = close / df["sma_50"]
    df["price_to_sma200"] = close / df["sma_200"]

    # ── Volatility ────────────────────────────────────────────────────────────
    df["volatility_5"]  = df["return_1d"].rolling(5).std()
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    df["volatility_20"] = df["return_1d"].rolling(20).std()

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    df["RSI_14"] = _rsi(close, 14)
    df["RSI_7"]  = _rsi(close, 7)

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / bb_mid
    df["BB_pct"]   = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)

    # ── ATR (14) ─────────────────────────────────────────────────────────────
    df["ATR_14"] = _atr(high, low, close, 14)
    df["ATR_pct"] = df["ATR_14"] / close  # Normalised ATR

    # ── Volume features ───────────────────────────────────────────────────────
    df["volume_sma20"]   = vol.rolling(20).mean()
    df["volume_change"]  = vol.pct_change(1)
    df["volume_ratio"]   = vol / (df["volume_sma20"] + 1)

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["momentum_10"] = close / close.shift(10) - 1
    df["momentum_20"] = close / close.shift(20) - 1

    # ── Stochastic Oscillator ─────────────────────────────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── Target: next-day close (what we are predicting) ──────────────────────
    df["Target"] = close.shift(-1)

    # Drop rows with NaN (from rolling windows and target shift)
    df = df.dropna()

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Returns list of feature column names (excludes OHLCV raw + Target)."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "Target"}
    return [c for c in df.columns if c not in exclude]


# ── Private helpers ───────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
