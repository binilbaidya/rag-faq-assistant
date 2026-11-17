from __future__ import annotations

import base64
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo.errors import PyMongoError

from app.config import get_settings
from app.database.mongo_history import get_mongo_client
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatMessage,
    ChatRole,
    ConversationSummary,
    ConversationsResponse,
    HistoryPagination,
)


logger = logging.getLogger(__name__)

_TITLE_MAX_LEN = 60
_UNTITLED = {"$or": [{"title": {"$exists": False}}, {"title": ""}, {"title": None}]}
_MAX_EMPTY_SCAN = 20


class ConversationNotFound(LookupError):
    """The id is unknown, or belongs to another user. Never say which."""


class ConversationIdRequired(LookupError):
    """An authenticated turn arrived without a conversation to attach it to."""


class InvalidCursor(LookupError):
    """The `before` cursor could not be decoded."""


class ChatHistoryUnavailable(RuntimeError):
    """Mongo rejected a read/write. Callers surface 503 rather than pretending it succeeded."""


class ChatHistoryService:
    """Thread metadata in chat_conversations; message rows in chat_messages."""

    def __init__(self) -> None:
        settings = get_settings()
        db = get_mongo_client()[settings.mongo_db]
        self.conversations = db[settings.mongo_conversations_collection]
        self.messages = db[settings.mongo_chat_collection]
        self._ensure_indexes()

    def save_message(
        self,
        user_uuid: str,
        role: ChatRole,
        message: str,
        *,
        conversation_id: str,
        sources: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Persist a message. Returns the new conversation title when this user message set it."""
        now = datetime.now(timezone.utc)
        text = message.strip()
        try:
            self.messages.insert_one(
                {
                    "user_uuid": user_uuid,
                    "conversation_id": conversation_id,
                    "role": role,
                    "message": text,
                    "sources": sources or [],
                    "metadata": metadata or {},
                    "timestamp": now,
                }
            )
            filt = {"user_uuid": user_uuid, "conversation_id": conversation_id}
            touch = {"updated_at": now}
            if role == "user":
                provisional = title_from_message(text)
                titled = self.conversations.update_one(
                    {**filt, **_UNTITLED},
                    {"$set": {**touch, "title": provisional}},
                )
                if titled.matched_count:
                    return provisional or None
            self.conversations.update_one(filt, {"$set": touch})
        except PyMongoError as exc:
            logger.exception("Failed to save chat message for user %s", user_uuid)
            raise ChatHistoryUnavailable("Chat history is temporarily unavailable.") from exc
        return None

    def set_title(self, user_uuid: str, conversation_id: str, title: str) -> str | None:
        text = " ".join(title.split()).strip("\"'`“”‘’")
        text = text.rstrip(".!?").strip()
        if not text:
            return None
        text = _truncate_title(text)
        try:
            self.conversations.update_one(
                {"user_uuid": user_uuid, "conversation_id": conversation_id},
                {"$set": {"title": text}},
            )
            return text
        except Exception:
            logger.exception(
                "Failed to set title for conversation %s user %s",
                conversation_id,
                user_uuid,
            )
            return None

    def get_messages(
        self,
        user_uuid: str,
        *,
        conversation_id: str,
        limit: int = 50,
        before: str | None = None,
    ) -> ChatHistoryResponse:
        """Newest page first; pass the returned next_cursor as `before` to walk backwards."""
        self.require_owned(user_uuid, conversation_id)
        rows = self._fetch_page(user_uuid, conversation_id, limit=limit + 1, before=_decode_cursor(before))
        has_more = len(rows) > limit
        rows = rows[:limit]
        return ChatHistoryResponse(
            user_uuid=user_uuid,
            conversation_id=conversation_id,
            messages=[_to_message(doc) for doc in reversed(rows)],
            pagination=HistoryPagination(
                limit=limit,
                has_more=has_more,
                next_cursor=_encode_cursor(rows[-1]) if has_more and rows else None,
            ),
        )

    def get_recent_messages(
        self,
        user_uuid: str,
        *,
        conversation_id: str,
        limit: int = 12,
    ) -> list[ChatMessage]:
        rows = self._fetch_page(user_uuid, conversation_id, limit=max(1, limit), before=None)
        return [_to_message(doc) for doc in reversed(rows)]

    def require_owned(self, user_uuid: str, conversation_id: str) -> None:
        if not self._owns(user_uuid, conversation_id):
            raise ConversationNotFound("Conversation not found.")

    def new_conversation(self, user_uuid: str) -> str:
        """Reuse an empty thread if one exists; otherwise create a new id."""
        existing = self._empty_conversation_id(user_uuid)
        if existing:
            try:
                self.conversations.update_one(
                    {"user_uuid": user_uuid, "conversation_id": existing},
                    {"$set": {"updated_at": datetime.now(timezone.utc)}},
                )
            except Exception:
                logger.exception("Failed to touch empty conversation for user %s", user_uuid)
            return existing

        conversation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        try:
            self.conversations.insert_one(
                {
                    "conversation_id": conversation_id,
                    "user_uuid": user_uuid,
                    "title": "",
                    "created_at": now,
                    "updated_at": now,
                }
            )
        except PyMongoError as exc:
            logger.exception("Failed to create conversation for user %s", user_uuid)
            raise ChatHistoryUnavailable("Chat history is temporarily unavailable.") from exc
        return conversation_id

    def list_conversations(self, user_uuid: str, *, limit: int = 50) -> ConversationsResponse:
        conversations: list[ConversationSummary] = []
        try:
            cursor = (
                self.conversations.find(
                    {
                        "user_uuid": user_uuid,
                        "title": {"$exists": True, "$nin": ["", None]},
                    },
                    {"_id": 0, "conversation_id": 1, "title": 1, "updated_at": 1},
                )
                .sort("updated_at", -1)
                .limit(max(1, limit))
            )
            conversations = [
                ConversationSummary(
                    conversation_id=doc["conversation_id"],
                    title=doc["title"],
                    updated_at=doc["updated_at"],
                )
                for doc in cursor
            ]
        except PyMongoError as exc:
            logger.exception("Failed to list conversations for user %s", user_uuid)
            raise ChatHistoryUnavailable("Chat history is temporarily unavailable.") from exc
        return ConversationsResponse(user_uuid=user_uuid, conversations=conversations)

    def _empty_conversation_id(self, user_uuid: str) -> str | None:
        """Newest thread for this user that still has no message rows."""
        try:
            candidates = [
                doc["conversation_id"]
                for doc in self.conversations.find(
                    {"user_uuid": user_uuid, **_UNTITLED},
                    {"_id": 0, "conversation_id": 1},
                )
                .sort("updated_at", -1)
                .limit(_MAX_EMPTY_SCAN)
                if str(doc.get("conversation_id") or "").strip()
            ]
            if not candidates:
                return None
            used = set(
                self.messages.distinct(
                    "conversation_id",
                    {"user_uuid": user_uuid, "conversation_id": {"$in": candidates}},
                )
            )
            return next((item for item in candidates if item not in used), None)
        except PyMongoError:
            logger.warning("Failed to look up empty conversation for %s", user_uuid, exc_info=True)
            return None

    def _owns(self, user_uuid: str, conversation_id: str) -> bool:
        try:
            return (
                self.conversations.find_one(
                    {"user_uuid": user_uuid, "conversation_id": conversation_id},
                    {"_id": 1},
                )
                is not None
            )
        except PyMongoError as exc:
            logger.exception("Failed to verify conversation %s for user %s", conversation_id, user_uuid)
            raise ChatHistoryUnavailable("Chat history is temporarily unavailable.") from exc

    def _fetch_page(
        self,
        user_uuid: str,
        conversation_id: str,
        *,
        limit: int,
        before: tuple[datetime, ObjectId] | None,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"user_uuid": user_uuid, "conversation_id": conversation_id}
        if before is not None:
            timestamp, message_id = before
            query["$or"] = [
                {"timestamp": {"$lt": timestamp}},
                {"timestamp": timestamp, "_id": {"$lt": message_id}},
            ]
        try:
            return list(
                self.messages.find(query).sort([("timestamp", -1), ("_id", -1)]).limit(limit)
            )
        except PyMongoError as exc:
            logger.exception("Failed to fetch chat history for user %s", user_uuid)
            raise ChatHistoryUnavailable("Chat history is temporarily unavailable.") from exc

    def _ensure_indexes(self) -> None:
        try:
            self.conversations.create_index("conversation_id", unique=True)
            self.conversations.create_index([("user_uuid", 1), ("updated_at", -1)])
            self.messages.create_index([("user_uuid", 1), ("conversation_id", 1), ("timestamp", 1)])
        except PyMongoError:
            logger.warning("Could not create chat history indexes", exc_info=True)


def _encode_cursor(doc: dict[str, Any]) -> str:
    raw = f"{doc['timestamp'].isoformat()}|{doc['_id']}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _decode_cursor(cursor: str | None) -> tuple[datetime, ObjectId] | None:
    if not cursor:
        return None
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        timestamp, separator, message_id = raw.partition("|")
        if not separator:
            raise ValueError("missing separator")
        return datetime.fromisoformat(timestamp), ObjectId(message_id)
    except (ValueError, InvalidId, UnicodeDecodeError, TypeError) as exc:
        raise InvalidCursor("Invalid pagination cursor.") from exc


def title_from_message(message: str) -> str:
    return _truncate_title(" ".join(message.split()))


def _truncate_title(text: str) -> str:
    if len(text) <= _TITLE_MAX_LEN:
        return text
    return f"{text[: _TITLE_MAX_LEN - 1].rstrip()}…"


def _to_message(doc: dict[str, Any]) -> ChatMessage:
    metadata = doc.get("metadata")
    return ChatMessage(
        role=doc["role"],
        message=doc["message"],
        sources=[item for item in (doc.get("sources") or []) if isinstance(item, dict)],
        metadata=metadata if isinstance(metadata, dict) else {},
        timestamp=doc["timestamp"],
    )
