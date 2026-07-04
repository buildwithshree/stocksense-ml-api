import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all technical indicator features to OHLCV DataFrame.
    Input: DataFrame with columns Open, High, Low, Close, Volume
    Output: DataFrame with all features + two target columns:
        - Target        : next-day absolute close (kept for reporting only,
                           NEVER used as the model's training target)
        - Target_return : next-day percentage return (close_t+1 / close_t - 1)
                           — THIS is what the model actually trains on.

    Why: absolute price is non-stationary. AAPL traded ~$10 in 2007 and
    ~$300 in 2026 — a model trained to predict absolute dollar price across
    that span is really just memorizing "which era produces which price
    range," and collapses the moment it's asked to extrapolate beyond the
    price levels it saw in training. Returns are scale-invariant: a 1%
    move means the same thing whether the stock is at $10 or $300, so a
    model trained on returns generalizes across price regimes instead of
    breaking the moment the stock moves into new territory.

    For the same reason, every ABSOLUTE-price-level feature below (raw
    sma_w, ema_w, MACD, MACD_signal, MACD_hist, BB_upper, BB_lower,
    ATR_14) is computed here because downstream normalized versions are
    derived from them, but get_feature_columns() EXCLUDES the raw
    absolute versions from what's actually fed to the model — only the
    normalized (ratio/percentage) versions are used as model inputs.
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
    # Raw sma_w / ema_w are absolute price levels — kept only as intermediate
    # values to compute the normalized ratios below. Excluded from feature_cols.
    for w in [5, 10, 20, 50, 200]:
        df[f"sma_{w}"] = close.rolling(w).mean()
        df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

    # Price relative to MAs (normalised, scale-invariant — these ARE features)
    df["price_to_sma5"]   = close / df["sma_5"]
    df["price_to_sma10"]  = close / df["sma_10"]
    df["price_to_sma20"]  = close / df["sma_20"]
    df["price_to_sma50"]  = close / df["sma_50"]
    df["price_to_sma200"] = close / df["sma_200"]
    df["price_to_ema5"]   = close / df["ema_5"]
    df["price_to_ema10"]  = close / df["ema_10"]
    df["price_to_ema20"]  = close / df["ema_20"]
    df["price_to_ema50"]  = close / df["ema_50"]
    df["price_to_ema200"] = close / df["ema_200"]

    # ── Volatility (already stationary — based on returns, not price) ────────
    df["volatility_5"]  = df["return_1d"].rolling(5).std()
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    df["volatility_20"] = df["return_1d"].rolling(20).std()

    # ── RSI (14) — already bounded 0-100, stationary regardless of price ─────
    df["RSI_14"] = _rsi(close, 14)
    df["RSI_7"]  = _rsi(close, 7)

    # ── MACD — raw values are absolute price-level; normalize by price ───────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26                          # excluded from features
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()   # excluded
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]         # excluded
    df["MACD_pct"]        = df["MACD"] / close                 # feature
    df["MACD_signal_pct"] = df["MACD_signal"] / close          # feature
    df["MACD_hist_pct"]   = df["MACD_hist"] / close            # feature

    # ── Bollinger Bands — BB_upper/BB_lower are absolute; width/pct already
    #    normalized, kept as-is ───────────────────────────────────────────────
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std                       # excluded
    df["BB_lower"] = bb_mid - 2 * bb_std                       # excluded
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / bb_mid
    df["BB_pct"]   = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)

    # ── ATR (14) — raw is absolute price-range; ATR_pct already normalized ───
    df["ATR_14"] = _atr(high, low, close, 14)                 # excluded
    df["ATR_pct"] = df["ATR_14"] / close                       # feature

    # ── Volume features (already relative/ratio-based) ───────────────────────
    df["volume_sma20"]   = vol.rolling(20).mean()
    df["volume_change"]  = vol.pct_change(1)
    df["volume_ratio"]   = vol / (df["volume_sma20"] + 1)

    # ── Momentum (already percentage-based) ───────────────────────────────────
    df["momentum_10"] = close / close.shift(10) - 1
    df["momentum_20"] = close / close.shift(20) - 1

    # ── Stochastic Oscillator (already bounded 0-100) ─────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── Targets ────────────────────────────────────────────────────────────
    # Target: absolute next-day close — kept ONLY for human-readable reporting/
    # debugging. NEVER pass this to a model as the training label.
    df["Target"] = close.shift(-1)

    # Target_return: what the model actually trains on. Stationary regardless
    # of what decade or price regime the row comes from.
    df["Target_return"] = close.shift(-1) / close - 1

    # Drop rows with NaN (from rolling windows and target shift)
    df = df.dropna()

    return df


# Absolute-price-level columns that must NEVER be fed to a model trained
# across multiple price regimes — kept in the DataFrame only as intermediate
# values for computing the normalized features above.
_ABSOLUTE_PRICE_COLUMNS = {
    "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower",
    "ATR_14",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Returns list of feature column names actually fed to the model.
    Excludes: raw OHLCV, both target columns, and every absolute-price-level
    intermediate column (see _ABSOLUTE_PRICE_COLUMNS) — only normalized,
    scale-invariant features are used as model inputs.
    """
    exclude = {"Open", "High", "Low", "Close", "Volume", "Target", "Target_return"}
    exclude |= _ABSOLUTE_PRICE_COLUMNS
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