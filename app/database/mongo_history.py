from __future__ import annotations

from functools import lru_cache

from pymongo import MongoClient

from app.config import get_settings


MONGO_TIMEOUT_MS = 1000
MONGO_SOCKET_TIMEOUT_MS = 5000


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    settings = get_settings()
    return MongoClient(
        settings.mongo_uri,
        serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
        connectTimeoutMS=MONGO_TIMEOUT_MS,
        socketTimeoutMS=MONGO_SOCKET_TIMEOUT_MS,
        waitQueueTimeoutMS=MONGO_TIMEOUT_MS,
        maxPoolSize=20,
        retryWrites=True,
        tz_aware=True,
    )
