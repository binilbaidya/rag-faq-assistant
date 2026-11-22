from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=4000)


class ChatTurnRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)
    conversation_id: str | None = Field(default=None, max_length=128)


class ChatTurnResult(BaseModel):
    message: str
    conversation_id: str | None = None
    conversation_title: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatStreamEvent(BaseModel):
    event: Literal["token", "done"]
    text: str | None = None
    message: str | None = None
    conversation_id: str | None = None
    conversation_title: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
