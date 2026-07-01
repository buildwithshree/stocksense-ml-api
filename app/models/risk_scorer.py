import numpy as np
import pandas as pd


def compute_risk_score(df: pd.DataFrame, predicted_close: float, last_close: float) -> tuple[int, str]:
    """
    Mathematical risk score formula (0–100, higher = riskier).
    Fully explainable in viva/interview.

    Components:
        0.30 × volatility_score    — annualised 20d return std
        0.25 × drawdown_score      — max drawdown over 252 days
        0.20 × uncertainty_score   — prediction confidence gap
        0.15 × atr_score           — ATR as % of price (range risk)
        0.10 × volume_risk_score   — volume instability
    """
    close = df["Close"]

    # ── 1. Volatility score (annualised) ─────────────────────────────────────
    returns   = close.pct_change().dropna()
    vol_20d   = returns.rolling(20).std().iloc[-1]
    annual_vol = vol_20d * np.sqrt(252)
    # Normalise: 0% vol → 0, 80%+ vol → 100
    volatility_score = float(np.clip(annual_vol / 0.80 * 100, 0, 100))

    # ── 2. Max drawdown score ─────────────────────────────────────────────────
    window = min(252, len(close))
    roll_max   = close.rolling(window).max()
    drawdown   = (close - roll_max) / roll_max
    max_dd     = abs(drawdown.min())
    # 0% drawdown → 0, 50%+ drawdown → 100
    drawdown_score = float(np.clip(max_dd / 0.50 * 100, 0, 100))

    # ── 3. Uncertainty score — prediction gap ────────────────────────────────
    pred_change_pct = abs((predicted_close - last_close) / last_close)
    # Larger predicted move = more uncertainty
    uncertainty_score = float(np.clip(pred_change_pct / 0.10 * 100, 0, 100))

    # ── 4. ATR score (ATR as % of price) ─────────────────────────────────────
    high, low = df["High"], df["Low"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14    = tr.rolling(14).mean().iloc[-1]
    atr_pct   = atr_14 / last_close
    # 0% ATR → 0, 5%+ ATR → 100
    atr_score = float(np.clip(atr_pct / 0.05 * 100, 0, 100))

    # ── 5. Volume instability ────────────────────────────────────────────────
    vol      = df["Volume"]
    vol_cv   = vol.rolling(20).std().iloc[-1] / (vol.rolling(20).mean().iloc[-1] + 1)
    # CV > 1.5 = highly unstable
    volume_risk_score = float(np.clip(vol_cv / 1.5 * 100, 0, 100))

    # ── Weighted sum ──────────────────────────────────────────────────────────
    raw_score = (
        0.30 * volatility_score +
        0.25 * drawdown_score   +
        0.20 * uncertainty_score +
        0.15 * atr_score        +
        0.10 * volume_risk_score
    )
    risk_score = int(np.clip(round(raw_score), 0, 100))

    # ── Label ─────────────────────────────────────────────────────────────────
    if risk_score < 30:
        label = "Low"
    elif risk_score < 55:
        label = "Moderate"
    elif risk_score < 75:
        label = "High"
    else:
        label = "Very High"

    return risk_score, label
