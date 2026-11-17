from functools import lru_cache

from app.agents.faq_agent import FAQAgent
from app.agents.orchestrator import ChatOrchestrator
from app.config import get_settings
from app.database.ai_settings import AiSettingsService
from app.database.chat_history import ChatHistoryService
from app.database.vector_store import FAQVectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> FAQVectorStore:
    return FAQVectorStore()


@lru_cache(maxsize=1)
def get_faq_agent() -> FAQAgent:
    return FAQAgent(vector_store=get_vector_store())


@lru_cache(maxsize=1)
def get_chat_history() -> ChatHistoryService | None:
    return ChatHistoryService() if get_settings().mongo_uri else None


@lru_cache(maxsize=1)
def get_ai_settings() -> AiSettingsService:
    return AiSettingsService()


@lru_cache(maxsize=1)
def get_chat_orchestrator() -> ChatOrchestrator:
    return ChatOrchestrator(
        faq_agent=get_faq_agent(),
        chat_history=get_chat_history(),
        ai_settings=get_ai_settings(),
    )
