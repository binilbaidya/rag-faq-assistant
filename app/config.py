from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypeVar
from urllib.parse import quote_plus

from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", int, float)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _str_env(name: str, default: str) -> str:
    """A set-but-empty var (as shipped in .env.example) must fall back, not return ''."""
    return os.getenv(name, "").strip() or default


def _optional_env(name: str) -> str | None:
    return os.getenv(name, "").strip() or None


def _number_env(name: str, default: T, cast: Callable[[str], T]) -> T:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %r", name, raw, default)
        return default


def build_mongo_uri(
    *,
    host: str,
    port: int,
    db: str,
    user: str | None = None,
    password: str | None = None,
    auth_source: str = "admin",
) -> str:
    if user and password:
        return (
            f"mongodb://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port}/{db}?authSource={quote_plus(auth_source)}"
        )
    return f"mongodb://{host}:{port}/{db}"


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = field(default_factory=lambda: _str_env("APP_NAME", "FAQ RAG Assistant"))
    environment: str = field(default_factory=lambda: _str_env("ENVIRONMENT", "development"))
    log_level: str = field(default_factory=lambda: _str_env("LOG_LEVEL", "INFO").upper())
    openai_api_key: str | None = field(default_factory=lambda: _optional_env("OPENAI_API_KEY"))
    # Any OpenAI-compatible endpoint (e.g. Gemini); empty means api.openai.com.
    openai_base_url: str | None = field(default_factory=lambda: _optional_env("OPENAI_BASE_URL"))
    openai_chat_model: str = field(default_factory=lambda: _str_env("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_embedding_model: str = field(default_factory=lambda: _str_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    jwt_secret_key: str | None = field(default_factory=lambda: _optional_env("JWT_SECRET_KEY"))
    # Full URI wins (needed for Atlas mongodb+srv://); otherwise built from host/port/user.
    mongo_uri_override: str | None = field(default_factory=lambda: _optional_env("MONGO_URI"))
    mongo_host: str = field(default_factory=lambda: _str_env("MONGO_HOST", "127.0.0.1"))
    mongo_port: int = field(default_factory=lambda: _number_env("MONGO_PORT", 27017, int))
    mongo_user: str | None = field(default_factory=lambda: _optional_env("MONGO_USER"))
    mongo_password: str | None = field(default_factory=lambda: _optional_env("MONGO_PASSWORD"))
    mongo_db: str = field(default_factory=lambda: _str_env("MONGO_DB", "faq_bot"))
    mongo_chat_collection: str = field(default_factory=lambda: _str_env("MONGO_CHAT_COLLECTION", "chat_messages"))
    mongo_conversations_collection: str = field(
        default_factory=lambda: _str_env("MONGO_CONVERSATIONS_COLLECTION", "chat_conversations")
    )
    qdrant_url: str = field(default_factory=lambda: _str_env("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: _optional_env("QDRANT_API_KEY"))
    qdrant_collection: str = field(default_factory=lambda: _str_env("QDRANT_COLLECTION", "faqs"))
    llm_timeout_seconds: float = field(default_factory=lambda: _number_env("LLM_TIMEOUT_SECONDS", 30.0, float))
    vector_size: int = field(default_factory=lambda: _number_env("VECTOR_SIZE", 1536, int))
    store_id: str = field(default_factory=lambda: _str_env("STORE_ID", "faq"))
    cors_allow_all: bool = field(default_factory=lambda: _bool_env("CORS_ALLOW_ALL", default=False))
    cors_allowed_origins: tuple[str, ...] = field(default_factory=lambda: _csv_env("CORS_ALLOWED_ORIGINS"))
    chat_rate_limit: int = field(default_factory=lambda: _number_env("CHAT_RATE_LIMIT", 20, int))
    chat_rate_window_seconds: float = field(default_factory=lambda: _number_env("CHAT_RATE_WINDOW_SECONDS", 60.0, float))
    ai_default_enabled: bool = field(default_factory=lambda: _bool_env("AI_DEFAULT_ENABLED", default=True))
    ai_force_disabled: bool = field(default_factory=lambda: _bool_env("AI_FORCE_DISABLED", default=False))

    @property
    def mongo_uri(self) -> str:
        return self.mongo_uri_override or build_mongo_uri(
            host=self.mongo_host,
            port=self.mongo_port,
            db=self.mongo_db,
            user=self.mongo_user,
            password=self.mongo_password,
            auth_source="admin",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
