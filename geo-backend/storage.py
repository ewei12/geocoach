import os
import time
import json
from abc import ABC, abstractmethod

DEFAULT_TTL_SECONDS = 600  # 10 minutes


class PendingStore(ABC):
    @abstractmethod
    def set(self, request_id, data): ...

    @abstractmethod
    def pop(self, request_id): ...


class InMemoryPendingStore(PendingStore):
    """Single-process store. Entries carry an expiry timestamp;
    expired entries are swept lazily on access."""

    def __init__(self, ttl_seconds=DEFAULT_TTL_SECONDS):
        self._data = {}
        self._ttl = ttl_seconds

    def _is_expired(self, entry):
        return time.time() > entry["_expires_at"]

    def set(self, request_id, data):
        self._data[request_id] = {**data, "_expires_at": time.time() + self._ttl}

    def pop(self, request_id):
        entry = self._data.pop(request_id, None)
        if entry is None:
            return None
        if self._is_expired(entry):
            return None
        entry.pop("_expires_at", None)
        return entry


class RedisPendingStore(PendingStore):
    """Multi-process/multi-instance store. Requires redis-py: pip install redis"""

    def __init__(self, redis_url=None, ttl_seconds=DEFAULT_TTL_SECONDS):
        import redis  # imported lazily so the in-memory path never needs the dep
        self._client = redis.from_url(redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self._ttl = ttl_seconds

    def _serialize(self, data):
        data = dict(data)
        if "embedding" in data and hasattr(data["embedding"], "tolist"):
            data["embedding"] = data["embedding"].tolist()
        return json.dumps(data)

    def _deserialize(self, raw):
        import numpy as np
        data = json.loads(raw)
        if "embedding" in data:
            data["embedding"] = np.array(data["embedding"])
        return data

    def set(self, request_id, data):
        self._client.setex(request_id, self._ttl, self._serialize(data))

    def pop(self, request_id):
        pipe = self._client.pipeline()
        pipe.get(request_id)
        pipe.delete(request_id)
        raw, _ = pipe.execute()
        if raw is None:
            return None
        return self._deserialize(raw)


def get_pending_store():
    """
    Backend selection:
      PENDING_BACKEND=memory -> force in-memory
      PENDING_BACKEND=redis  -> force Redis (raises if unreachable)
    """
    backend = os.getenv("PENDING_BACKEND", "auto").lower()

    if backend == "memory":
        return InMemoryPendingStore()

    if backend == "redis":
        return RedisPendingStore()

    # detect automatically
    try:
        store = RedisPendingStore()
        store._client.ping()
        print("[storage] Redis detected — using RedisPendingStore", flush=True)
        return store
    except Exception as e:
        print(f"[storage] Redis unavailable ({e}) — falling back to InMemoryPendingStore", flush=True)
        return InMemoryPendingStore()