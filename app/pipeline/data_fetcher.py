import logging
import requests
import pandas as pd
from datetime import datetime
from typing import Tuple

from app.config import settings
from app.db.database import get_cached_ohlcv, write_ohlcv_cache

logger = logging.getLogger(__name__)

SUFFIX_CURRENCY = {".NS": "INR", ".BO": "INR"}

def detect_currency(ticker: str) -> str:
    for suffix, currency in SUFFIX_CURRENCY.items():
        if ticker.upper().endswith(suffix):
            return currency
    return "USD"

def fetch_ohlcv(ticker: str) -> Tuple[pd.DataFrame, str]:
    ticker = ticker.upper().strip()
    currency = detect_currency(ticker)

    # 1. Cache check
    cached = get_cached_ohlcv(ticker)
    if cached is not None and len(cached) >= 50:
        return cached, currency

    # 2. Alpha Vantage fetch
    # Strip exchange suffix for AV — it uses base symbol only
    av_symbol = ticker.split(".")[0]
    logger.info("Fetching OHLCV for %s (AV symbol: %s)", ticker, av_symbol)

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": av_symbol,
        "outputsize": "full",
        "apikey": settings.alpha_vantage_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ValueError(f"Alpha Vantage request failed for {ticker}: {e}")

    if "Note" in data:
        raise ValueError("Alpha Vantage rate limit hit. Try again in 1 minute.")

    if "Error Message" in data:
        raise ValueError(f"Invalid ticker {ticker}: {data['Error Message']}")

    ts = data.get("Time Series (Daily)")
    if not ts:
        raise ValueError(f"No data returned from Alpha Vantage for ticker: {ticker}")

    rows = []
    for date_str, vals in ts.items():
        rows.append({
            "Date": pd.to_datetime(date_str),
            "Open":   float(vals["1. open"]),
            "High":   float(vals["2. high"]),
            "Low":    float(vals["3. low"]),
            "Close":  float(vals["5. adjusted close"]),
            "Volume": int(vals["6. volume"]),
        })

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    df = df.dropna()

    if len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker}: only {len(df)} rows")

    write_ohlcv_cache(ticker, df, currency)
    logger.info("Fetched %d rows for %s via Alpha Vantage", len(df), ticker)
    return df, currency


def get_company_name(ticker: str) -> str:
    try:
        av_symbol = ticker.split(".")[0]
        url = "https://www.alphavantage.co/query"
        params = {"function": "OVERVIEW", "symbol": av_symbol, "apikey": settings.alpha_vantage_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        return data.get("Name") or ticker
    except Exception:
        return ticker