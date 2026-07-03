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

    Note: this cache is per-process and does NOT survive a Render restart.
    That's now handled separately by app.db.database.save_model_artifact /
    load_model_artifact — this in-memory layer is purely a fast path so a
    warm process doesn't hit Postgres on every request.
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


class TrainingRegistry:
    """
    Thread-safe registry of tickers currently being trained in the background.

    Problem this solves: if a ticker has no cached/persisted model and three
    users request it within the same few seconds, without this registry all
    three requests would each kick off their own ~minutes-long background
    training job — wasted CPU on a free-tier instance, and three separate
    writes racing on model_artifacts. With this registry, the first request
    claims the ticker and starts training; the other two get a 202 telling
    them training is already underway, instead of triggering duplicate work.
    """
    def __init__(self):
        self._in_progress: set[str] = set()
        self._lock = threading.Lock()

    def try_start(self, ticker: str) -> bool:
        """
        Atomically claim `ticker` for training.
        Returns True if this call claims it (caller should start training).
        Returns False if another request already claimed it (caller should
        not start a second training job).
        """
        with self._lock:
            if ticker in self._in_progress:
                return False
            self._in_progress.add(ticker)
            return True

    def finish(self, ticker: str) -> None:
        """Release the claim once training succeeds or fails. Always call
        this in a finally block — an un-released claim would wedge the
        ticker in a permanent 'training' state until process restart."""
        with self._lock:
            self._in_progress.discard(ticker)

    def is_in_progress(self, ticker: str) -> bool:
        with self._lock:
            return ticker in self._in_progress


# Singletons — imported everywhere
model_cache = ModelCache(stale_hours=24)
training_registry = TrainingRegistry()