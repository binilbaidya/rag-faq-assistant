from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel

from app.config import get_settings
from app.database.mongo_history import get_mongo_client


logger = logging.getLogger(__name__)

SETTINGS_DOCUMENT_ID = "global"
AI_DISABLED_MESSAGE = (
    "AI chat is currently unavailable. Please try again later or contact support."
)
_CACHE_TTL_SECONDS = 15


class AiSettings(BaseModel):
    ai_enabled: bool = True
    updated_at: datetime | None = None
    updated_by: str | None = None


class AiSettingsUpdate(BaseModel):
    ai_enabled: bool


class AiStatusResponse(BaseModel):
    ai_enabled: bool


class AiSettingsService:
    def __init__(self) -> None:
        settings = get_settings()
        self.collection = get_mongo_client()[settings.mongo_db]["ai_settings"]
        self._cached: dict[str, Any] | None = None
        self._cached_at = 0.0

    def is_ai_enabled(self) -> bool:
        return self._enabled_from(self._get_document())

    def get_settings(self) -> AiSettings:
        document = self._get_document()
        return AiSettings(
            ai_enabled=self._enabled_from(document),
            updated_at=(document or {}).get("updated_at"),
            updated_by=(document or {}).get("updated_by"),
        )

    @staticmethod
    def _enabled_from(document: dict[str, Any] | None) -> bool:
        settings = get_settings()
        if settings.ai_force_disabled:
            return False
        if document is None:
            return settings.ai_default_enabled
        return bool(document.get("ai_enabled", settings.ai_default_enabled))

    def set_ai_enabled(self, *, ai_enabled: bool, updated_by: str) -> AiSettings:
        now = datetime.now(timezone.utc)
        payload: dict[str, Any] = {
            "_id": SETTINGS_DOCUMENT_ID,
            "ai_enabled": ai_enabled,
            "updated_at": now,
            "updated_by": updated_by,
        }
        try:
            self.collection.replace_one({"_id": SETTINGS_DOCUMENT_ID}, payload, upsert=True)
        except Exception as exc:
            logger.exception("Failed to update AI settings")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Could not save AI settings to MongoDB. "
                    "Check MONGO_HOST/MONGO_PORT credentials and that Mongo is reachable from the bot."
                ),
            ) from exc
        self._cached, self._cached_at = payload, time.monotonic()
        return AiSettings(ai_enabled=ai_enabled, updated_at=now, updated_by=updated_by)

    def _get_document(self) -> dict[str, Any] | None:
        now = time.monotonic()
        if self._cached is not None and now - self._cached_at < _CACHE_TTL_SECONDS:
            return self._cached
        try:
            document = self.collection.find_one({"_id": SETTINGS_DOCUMENT_ID})
            self._cached = document if isinstance(document, dict) else None
            self._cached_at = now
            return self._cached
        except Exception:
            logger.exception("Failed to load AI settings")
            return None
