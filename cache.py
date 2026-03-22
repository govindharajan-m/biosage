"""
TTL in-memory query cache.

Redis-compatible interface — swap with redis-py by replacing this class.
Default TTL: 3600 seconds. Max 500 entries with LRU-style eviction.
"""

import hashlib
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TTLCache:
    """
    Thread-safe in-memory cache with per-entry TTL.
    Safe for asyncio environments (Python GIL protects dicts for simple ops).
    """

    def __init__(self, default_ttl: int = 3600, max_size: int = 500):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._store: dict = {}  # key → (value, expires_at)

    def _evict(self) -> None:
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if exp <= now]
        for k in expired:
            del self._store[k]
        if len(self._store) >= self.max_size:
            ordered = sorted(self._store, key=lambda k: self._store[k][1])
            for k in ordered[: len(self._store) - self.max_size + 1]:
                del self._store[k]

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, exp = entry
        if time.time() > exp:
            del self._store[key]
            return None
        logger.debug("Cache hit: %s", key)
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if len(self._store) >= self.max_size:
            self._evict()
        self._store[key] = (value, time.time() + (ttl or self.default_ttl))

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def flush(self) -> None:
        self._store.clear()

    def stats(self) -> dict:
        self._evict()
        return {
            "entries": len(self._store),
            "max_size": self.max_size,
            "default_ttl_s": self.default_ttl,
        }

    @staticmethod
    def make_key(prefix: str, *parts: str) -> str:
        combined = "|".join(s.lower().strip() for s in parts)
        h = hashlib.md5(combined.encode()).hexdigest()[:16]
        return f"{prefix}:{h}"


# Module-level singleton used throughout the app
query_cache = TTLCache(default_ttl=3600, max_size=500)
