import logging
import requests
import pandas as pd
from typing import Tuple

from app.config import settings
from app.db.database import get_cached_ohlcv, write_ohlcv_cache

logger = logging.getLogger(__name__)

SUFFIX_CURRENCY = {".NS": "INR", ".BO": "INR"}

BASE_URL = "https://api.twelvedata.com"


def detect_currency(ticker: str) -> str:
    for suffix, currency in SUFFIX_CURRENCY.items():
        if ticker.upper().endswith(suffix):
            return currency
    return "USD"


def _to_twelvedata_symbol(ticker: str) -> str:
    """
    Twelve Data uses plain symbols for US equities (AAPL) and
    'SYMBOL:EXCHANGE' or region suffixes for others. NSE/BSE tickers
    keep their .NS/.BO suffix stripped and pass exchange separately
    if ever needed — for now we only strip US-style suffixes since
    that's what the system currently supports (ticker.split('.')[0]
    mirrors the old Alpha Vantage behaviour so no caller code changes).
    """
    return ticker.split(".")[0]


def fetch_ohlcv(ticker: str) -> Tuple[pd.DataFrame, str]:
    ticker = ticker.upper().strip()
    currency = detect_currency(ticker)

    # 1. Cache check — unchanged, still the first line of defense against
    #    burning API quota on repeat requests for the same ticker.
    cached = get_cached_ohlcv(ticker)
    if cached is not None and len(cached) >= 50:
        return cached, currency

    td_symbol = _to_twelvedata_symbol(ticker)
    logger.info("Fetching OHLCV for %s (Twelve Data symbol: %s)", ticker, td_symbol)

    params = {
        "symbol": td_symbol,
        "interval": "1day",
        "outputsize": 5000,   # max allowed on free tier; daily interval
                              # returns full history since listing anyway
        "apikey": settings.twelve_data_key,
    }

    try:
        resp = requests.get(f"{BASE_URL}/time_series", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ValueError(f"Twelve Data request failed for {ticker}: {e}")

    # Twelve Data signals errors via status="error" + code/message —
    # NOT via HTTP status alone (a bad symbol still returns HTTP 200).
    if data.get("status") == "error":
        code = data.get("code")
        message = data.get("message", "Unknown error")
        if code == 429:
            raise ValueError(
                f"Twelve Data rate limit hit for {ticker}. Try again shortly."
            )
        raise ValueError(f"Twelve Data rejected request for {ticker}: {message}")

    values = data.get("values")
    if not values:
        raise ValueError(f"No data returned from Twelve Data for ticker: {ticker}")

    rows = []
    for entry in values:
        rows.append({
            "Date": pd.to_datetime(entry["datetime"]),
            "Open":   float(entry["open"]),
            "High":   float(entry["high"]),
            "Low":    float(entry["low"]),
            "Close":  float(entry["close"]),
            "Volume": int(float(entry["volume"])),  # some entries return "0.0"
        })

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]

    if len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker}: only {len(df)} rows")

    write_ohlcv_cache(ticker, df, currency)
    logger.info("Fetched %d rows for %s via Twelve Data", len(df), ticker)
    return df, currency


def get_company_name(ticker: str) -> str:
    """
    Uses Twelve Data's /quote endpoint (free tier) instead of Alpha
    Vantage's OVERVIEW. Keeping a single provider for the whole pipeline
    reduces the number of external dependencies that can independently
    break in production — worth it even though it costs one extra
    request per uncached ticker.
    """
    try:
        td_symbol = _to_twelvedata_symbol(ticker)
        resp = requests.get(
            f"{BASE_URL}/quote",
            params={"symbol": td_symbol, "apikey": settings.twelve_data_key},
            timeout=10,
        )
        data = resp.json()
        if data.get("status") == "error":
            return ticker
        return data.get("name") or ticker
    except Exception:
        return ticker