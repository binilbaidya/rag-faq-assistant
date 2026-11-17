from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ChatRole = Literal["user", "assistant"]


class V1ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(
        default=None,
        max_length=128,
        description="Required when authenticated. Omit for anonymous chat (no history).",
    )


class V1ChatResponse(BaseModel):
    message: str
    conversation_id: str | None = None
    conversation_title: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Timeline entry. Assistant rows mirror POST /v1/chat for UI replay."""

    role: ChatRole
    message: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


class HistoryPagination(BaseModel):
    limit: int
    has_more: bool = False
    next_cursor: str | None = Field(
        default=None,
        description="Pass as ?before= to load the page of older messages. Null when has_more is false.",
    )


class ChatHistoryResponse(BaseModel):
    user_uuid: str
    conversation_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    pagination: HistoryPagination = Field(default_factory=lambda: HistoryPagination(limit=0))


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    updated_at: datetime


class ConversationsResponse(BaseModel):
    user_uuid: str
    conversations: list[ConversationSummary] = Field(default_factory=list)


class ConversationIdResponse(BaseModel):
    conversation_id: str
