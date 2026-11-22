from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.models import HistoryTurn


ChatRole = Literal["user", "assistant"]


class V1ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(
        default=None,
        max_length=128,
        description="Required when authenticated. Anonymous clients may send a per-tab id to group logged chats.",
    )
    # Sent by the client so follow-ups work without a database; untrusted like the message itself.
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20, description="Last few turns, oldest first.")
    visitor_id: str | None = Field(default=None, max_length=64, description="Anonymous per-browser id that links a visitor's chats.")


class VisitorDetails(BaseModel):
    visitor_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


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
