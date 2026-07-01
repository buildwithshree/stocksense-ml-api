import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.models.selector import TrainResult

logger = logging.getLogger(__name__)


@dataclass
class CachedModel:
    result: TrainResult
    cached_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ModelCache:
    """
    Thread-safe in-memory cache for trained models.
    Hybrid strategy: serve cached model if fresh, retrain if stale.
    Stale threshold: 24 hours (matches stock_cache staleness).
    """
    def __init__(self, stale_hours: int = 24):
        self._cache: dict[str, CachedModel] = {}
        self._lock  = threading.Lock()
        self._stale_hours = stale_hours

    def get(self, ticker: str) -> Optional[TrainResult]:
        with self._lock:
            entry = self._cache.get(ticker)
            if entry is None:
                return None
            age_hours = (datetime.now(timezone.utc) - entry.cached_at).total_seconds() / 3600
            if age_hours > self._stale_hours:
                logger.info("Model cache STALE for %s (%.1fh old)", ticker, age_hours)
                del self._cache[ticker]
                return None
            logger.info("Model cache HIT for %s (%.1fh old)", ticker, age_hours)
            return entry.result

    def set(self, ticker: str, result: TrainResult) -> None:
        with self._lock:
            self._cache[ticker] = CachedModel(result=result)
            logger.info("Model cache SET for %s (%s)", ticker, result.model_name)

    def invalidate(self, ticker: str) -> None:
        with self._lock:
            self._cache.pop(ticker, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# Singleton — imported everywhere
model_cache = ModelCache(stale_hours=24)
