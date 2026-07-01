import logging
from typing import Tuple

import pandas as pd
import yfinance as yf

from app.config import settings
from app.db.database import get_cached_ohlcv, write_ohlcv_cache

logger = logging.getLogger(__name__)

# Currency mapping by suffix — used to set correct currency in stock_cache
SUFFIX_CURRENCY = {
    ".NS": "INR",  # NSE India
    ".BO": "INR",  # BSE India
}


def detect_currency(ticker: str) -> str:
    for suffix, currency in SUFFIX_CURRENCY.items():
        if ticker.upper().endswith(suffix):
            return currency
    return "USD"


def fetch_ohlcv(ticker: str) -> Tuple[pd.DataFrame, str]:
    """
    Returns (OHLCV DataFrame, currency).
    Strategy:
      1. Check stock_cache — return if fresh
      2. Fetch from yfinance — write to cache — return
    Raises ValueError if ticker is invalid or insufficient data.
    """
    ticker = ticker.upper()
    currency = detect_currency(ticker)

    # 1. Cache check
    cached = get_cached_ohlcv(ticker)
    if cached is not None and len(cached) >= 50:
        return cached, currency

    # 2. Live fetch
    logger.info("Fetching live OHLCV for %s from yfinance", ticker)
    try:
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(
            period=f"{settings.default_period_years}y",
            interval="1d",
            auto_adjust=True,
            actions=False,
        )
    except Exception as e:
        raise ValueError(f"yfinance fetch failed for {ticker}: {e}")

    if df is None or df.empty:
        raise ValueError(f"No data returned from yfinance for ticker: {ticker}")

    # yfinance returns timezone-aware index — normalise to date only for DB
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker}: only {len(df)} rows")

    # 3. Write to cache
    write_ohlcv_cache(ticker, df, currency)

    logger.info("Fetched %d rows for %s", len(df), ticker)
    return df, currency


def get_company_name(ticker: str) -> str:
    """Best-effort company name fetch. Falls back to ticker string."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker
