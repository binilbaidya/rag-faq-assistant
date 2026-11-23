from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypeVar

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
    # Optional: chat history and the runtime AI switch. Empty runs without MongoDB.
    mongo_uri: str | None = field(default_factory=lambda: _optional_env("MONGO_URI"))
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
    # A persona name from faq_agent.PERSONAS (polite, fun, professional) or custom text; safety rules stay fixed.
    assistant_persona: str = field(default_factory=lambda: _str_env("ASSISTANT_PERSONA", "polite"))
    # Exact reply when the FAQ has no answer; give it your own tone per deployment.
    no_answer_message: str = field(
        default_factory=lambda: _str_env("NO_ANSWER_MESSAGE", "I'm sorry, I don't have the answer to that question at the moment.")
    )
    # Saves anonymous visitors' chats to MongoDB (needs MONGO_URI) so the owner can read them; no IP is stored.
    log_anonymous_chats: bool = field(default_factory=lambda: _bool_env("LOG_ANONYMOUS_CHATS", default=False))
    # Lets the bot chat and joke a little beyond the FAQ (never inventing facts); off keeps it strictly grounded.
    allow_small_talk: bool = field(default_factory=lambda: _bool_env("ALLOW_SMALL_TALK", default=False))
    cors_allow_all: bool = field(default_factory=lambda: _bool_env("CORS_ALLOW_ALL", default=False))
    cors_allowed_origins: tuple[str, ...] = field(default_factory=lambda: _csv_env("CORS_ALLOWED_ORIGINS"))
    # Where the visitor's IP comes from: "x-forwarded-for", a CDN header such as "cf-connecting-ip", or "none" (direct connection).
    client_ip_header: str = field(default_factory=lambda: _str_env("CLIENT_IP_HEADER", "x-forwarded-for").lower())
    chat_rate_limit: int = field(default_factory=lambda: _number_env("CHAT_RATE_LIMIT", 20, int))
    chat_rate_window_seconds: float = field(default_factory=lambda: _number_env("CHAT_RATE_WINDOW_SECONDS", 60.0, float))
    # Site-wide cap across all visitors, so many IPs can't drain a free LLM quota.
    chat_daily_limit: int = field(default_factory=lambda: _number_env("CHAT_DAILY_LIMIT", 500, int))
    ai_default_enabled: bool = field(default_factory=lambda: _bool_env("AI_DEFAULT_ENABLED", default=True))
    ai_force_disabled: bool = field(default_factory=lambda: _bool_env("AI_FORCE_DISABLED", default=False))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
