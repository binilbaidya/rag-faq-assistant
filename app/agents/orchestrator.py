from __future__ import annotations

from collections.abc import AsyncIterator

from starlette.concurrency import run_in_threadpool

from app.agents.faq_agent import FAQAgent, FAQRequest
from app.agents.models import ChatStreamEvent, ChatTurnRequest, ChatTurnResult
from app.auth import AuthenticatedUser
from app.config import get_settings
from app.database.ai_settings import AI_DISABLED_MESSAGE, AiSettingsService
from app.database.chat_history import ChatHistoryService, ConversationIdRequired


class ChatOrchestrator:
    """One chat turn: AI switch check, history bookkeeping for signed-in users, then the FAQ answer.

    chat_history is None when MongoDB isn't configured; every turn is then answered without history.
    """

    def __init__(
        self,
        faq_agent: FAQAgent | None = None,
        chat_history: ChatHistoryService | None = None,
        ai_settings: AiSettingsService | None = None,
    ) -> None:
        self.faq_agent = faq_agent or FAQAgent()
        self.chat_history = chat_history
        self.ai_settings = ai_settings or AiSettingsService()

    async def run(
        self,
        user: AuthenticatedUser | None,
        request: ChatTurnRequest,
    ) -> ChatTurnResult:
        if not await run_in_threadpool(self.ai_settings.is_ai_enabled):
            return await self._disabled_result(user, request)

        conversation_id, conversation_title = await self._begin(user, request)
        answer = await self.faq_agent.answer(FAQRequest(question=request.message, history=request.history))
        result = ChatTurnResult(message=answer.answer, conversation_id=conversation_id, sources=answer.sources)
        return await self._finish(user, conversation_id, result, conversation_title)

    async def stream(
        self,
        user: AuthenticatedUser | None,
        request: ChatTurnRequest,
    ) -> AsyncIterator[ChatStreamEvent]:
        if not await run_in_threadpool(self.ai_settings.is_ai_enabled):
            result = await self._disabled_result(user, request)
            yield ChatStreamEvent(event="token", text=result.message)
            yield _done_event(result)
            return

        conversation_id, conversation_title = await self._begin(user, request)
        faq_request = FAQRequest(question=request.message, history=request.history)
        results = await run_in_threadpool(self.faq_agent.retrieve, faq_request)
        parts: list[str] = []
        async for chunk in self.faq_agent.stream_answer(faq_request, results):
            parts.append(chunk)
            yield ChatStreamEvent(event="token", text=chunk)
        result = ChatTurnResult(
            message="".join(parts),
            conversation_id=conversation_id,
            sources=self.faq_agent.sources(results),
        )
        result = await self._finish(user, conversation_id, result, conversation_title)
        yield _done_event(result)

    async def _begin(
        self,
        user: AuthenticatedUser | None,
        request: ChatTurnRequest,
    ) -> tuple[str | None, str | None]:
        """Saves the user's message; returns the conversation id and its title when this turn set it."""
        if user is None or self.chat_history is None:
            return None, None

        conversation_id = (request.conversation_id or "").strip()
        require_conversation_id(user, conversation_id)
        await run_in_threadpool(self.chat_history.require_owned, user.uuid, conversation_id)
        conversation_title = await run_in_threadpool(
            self.chat_history.save_message,
            user.uuid,
            "user",
            request.message,
            conversation_id=conversation_id,
        )
        return conversation_id, conversation_title

    async def _finish(
        self,
        user: AuthenticatedUser | None,
        conversation_id: str | None,
        result: ChatTurnResult,
        conversation_title: str | None,
    ) -> ChatTurnResult:
        if conversation_title:
            result = result.model_copy(update={"conversation_title": conversation_title})
        await self._save_assistant(user, conversation_id, result)
        return result

    async def _save_assistant(
        self,
        user: AuthenticatedUser | None,
        conversation_id: str | None,
        result: ChatTurnResult,
    ) -> None:
        if user is None or not conversation_id:
            return
        await run_in_threadpool(
            self.chat_history.save_message,
            user.uuid,
            "assistant",
            result.message,
            conversation_id=conversation_id,
            sources=result.sources,
            metadata=result.metadata,
        )

    async def _disabled_result(
        self,
        user: AuthenticatedUser | None,
        request: ChatTurnRequest,
    ) -> ChatTurnResult:
        conversation_id: str | None = None
        if user is not None and self.chat_history is not None:
            conversation_id = (request.conversation_id or "").strip() or None
            if conversation_id:
                await run_in_threadpool(self.chat_history.require_owned, user.uuid, conversation_id)
                await run_in_threadpool(
                    self.chat_history.save_message,
                    user.uuid,
                    "user",
                    request.message,
                    conversation_id=conversation_id,
                )
        result = ChatTurnResult(
            message=AI_DISABLED_MESSAGE,
            conversation_id=conversation_id,
            metadata={"ai_disabled": True},
        )
        await self._save_assistant(user, conversation_id, result)
        return result


def require_conversation_id(user: AuthenticatedUser | None, conversation_id: str | None) -> None:
    """Shared so routes can pre-flight it — a stream cannot change status once it has begun."""
    if user is not None and get_settings().mongo_uri and not (conversation_id or "").strip():
        raise ConversationIdRequired("conversation_id is required when authenticated.")


def _done_event(result: ChatTurnResult) -> ChatStreamEvent:
    return ChatStreamEvent(
        event="done",
        message=result.message,
        conversation_id=result.conversation_id,
        conversation_title=result.conversation_title,
        sources=result.sources,
        metadata=result.metadata,
    )
