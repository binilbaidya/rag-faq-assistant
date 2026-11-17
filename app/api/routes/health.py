from fastapi import APIRouter

from app.config import get_settings
from app.database.ai_settings import AiStatusResponse
from app.dependencies import get_ai_settings
from app.schemas.common import HealthResponse


router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name)


@router.get("/v1/ai/status", response_model=AiStatusResponse)
def ai_status() -> AiStatusResponse:
    return AiStatusResponse(ai_enabled=get_ai_settings().is_ai_enabled())
