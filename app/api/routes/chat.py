from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from fastapi.responses import StreamingResponse

from app.agents.models import ChatTurnRequest
from app.agents.orchestrator import ChatOrchestrator, require_conversation_id
from app.auth import AuthenticatedUser, get_current_user, get_optional_user
from app.database.chat_history import (
    ChatHistoryService,
    ChatHistoryUnavailable,
    ConversationIdRequired,
    InvalidCursor,
)
from app.dependencies import get_chat_history, get_chat_orchestrator
from app.config import get_settings
from app.rate_limit import client_ip, enforce_chat_rate_limit, enforce_client_rate_limit
from app.schemas.chat import (
    ChatHistoryResponse,
    ConversationIdResponse,
    ConversationsResponse,
    V1ChatRequest,
    VisitorDetails,
    V1ChatResponse,
)
from app.security import check_request


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.post("/conversations", response_model=ConversationIdResponse)
def start_conversation(
    user: AuthenticatedUser = Depends(get_current_user),
) -> ConversationIdResponse:
    with _storage_errors():
        return ConversationIdResponse(conversation_id=_history().new_conversation(user.uuid))


@router.get("/conversations", response_model=ConversationsResponse)
def list_conversations(
    user: AuthenticatedUser = Depends(get_current_user),
    limit: int = 50,
) -> ConversationsResponse:
    with _storage_errors():
        return _history().list_conversations(user.uuid, limit=_clamp(limit, 1, 200))


@router.get("/history", response_model=ChatHistoryResponse)
def chat_history(
    user: AuthenticatedUser = Depends(get_current_user),
    conversation_id: str = Query(min_length=1, max_length=128),
    limit: int = 50,
    before: str | None = Query(
        default=None,
        max_length=256,
        description="next_cursor from a previous response; omit for the newest page.",
    ),
) -> ChatHistoryResponse:
    try:
        with _storage_errors():
            return _history().get_messages(
                user.uuid,
                conversation_id=conversation_id,
                limit=_clamp(limit, 1, 200),
                before=before,
            )
    except LookupError as exc:
        raise _lookup_http_error(exc) from exc


@router.post("", response_model=V1ChatResponse, dependencies=[Depends(enforce_chat_rate_limit)])
async def v1_chat(
    payload: V1ChatRequest,
    request: Request,
    user: AuthenticatedUser | None = Depends(get_optional_user),
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator),
) -> V1ChatResponse:
    check_request(payload)
    try:
        require_conversation_id(user, payload.conversation_id)
        with _storage_errors():
            result = await orchestrator.run(user, _to_turn_request(payload, request))
    except LookupError as exc:
        raise _lookup_http_error(exc) from exc
    # Built from the model dump so a new ChatTurnResult field cannot be silently dropped.
    return V1ChatResponse.model_validate(result.model_dump())


@router.post("/stream", dependencies=[Depends(enforce_chat_rate_limit)])
async def v1_chat_stream(
    payload: V1ChatRequest,
    request: Request,
    user: AuthenticatedUser | None = Depends(get_optional_user),
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator),
) -> StreamingResponse:
    check_request(payload)
    try:
        require_conversation_id(user, payload.conversation_id)
    except LookupError as exc:
        raise _lookup_http_error(exc) from exc
    return StreamingResponse(
        _sse_from_orchestrator(orchestrator, user, _to_turn_request(payload, request)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/visitor", status_code=204, dependencies=[Depends(enforce_client_rate_limit)])
def save_visitor_details(payload: VisitorDetails) -> Response:
    """Name and email a visitor chose to share, linked to their chats by visitor_id."""
    check_request(payload)
    history = get_chat_history()
    if history is None or not get_settings().log_anonymous_chats:
        raise HTTPException(status_code=404, detail="Visitor details are not being collected.")
    with _storage_errors():
        history.identify_visitor(payload.visitor_id, payload.name, payload.email)
    return Response(status_code=204)


def _history() -> ChatHistoryService:
    history = get_chat_history()
    if history is None:
        raise HTTPException(status_code=503, detail="Chat history is disabled. Set MONGO_URI to enable it.")
    return history


@contextmanager
def _storage_errors() -> Iterator[None]:
    try:
        yield
    except ChatHistoryUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _to_turn_request(payload: V1ChatRequest, request: Request) -> ChatTurnRequest:
    return ChatTurnRequest(
        message=payload.message,
        history=payload.history,
        visitor_id=(payload.visitor_id or "").strip() or None,
        client_ip=client_ip(request),
        conversation_id=(payload.conversation_id or "").strip() or None,
    )


async def _sse_from_orchestrator(
    orchestrator: ChatOrchestrator,
    user: AuthenticatedUser | None,
    request: ChatTurnRequest,
) -> AsyncIterator[str]:
    try:
        async for event in orchestrator.stream(user, request):
            if event.event == "token" and event.text:
                yield _sse("token", {"message": event.text})
            elif event.event == "done":
                payload = event.model_dump(exclude={"event", "text"})
                yield _sse("done", {"ok": True, **payload})
    except LookupError as exc:
        yield _sse("error", {"message": _lookup_http_error(exc).detail})
    except ChatHistoryUnavailable as exc:
        logger.exception("Chat history unavailable during stream")
        yield _sse("error", {"message": str(exc)})
    except HTTPException as exc:
        yield _sse("error", {"message": str(exc.detail)})
    except Exception:
        logger.exception("Streaming chatbot response failed")
        yield _sse("error", {"message": "Request failed."})


def _lookup_http_error(exc: LookupError) -> HTTPException:
    if isinstance(exc, InvalidCursor):
        return HTTPException(status_code=400, detail="Invalid pagination cursor.")
    if isinstance(exc, ConversationIdRequired):
        return HTTPException(status_code=400, detail="conversation_id is required when authenticated.")
    return HTTPException(status_code=404, detail="Conversation not found.")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _clamp(value: int, low: int, high: int) -> int:
    return min(max(value, low), high)
