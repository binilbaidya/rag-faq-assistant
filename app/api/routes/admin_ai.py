from fastapi import APIRouter, Depends

from app.auth import AuthenticatedUser, require_superadmin_user
from app.database.ai_settings import AiSettings, AiSettingsUpdate
from app.dependencies import get_ai_settings


router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get("/ai", response_model=AiSettings)
def get_ai_settings_admin(
    _: AuthenticatedUser = Depends(require_superadmin_user),
) -> AiSettings:
    return get_ai_settings().get_settings()


@router.put("/ai", response_model=AiSettings)
def update_ai_settings(
    payload: AiSettingsUpdate,
    user: AuthenticatedUser = Depends(require_superadmin_user),
) -> AiSettings:
    return get_ai_settings().set_ai_enabled(ai_enabled=payload.ai_enabled, updated_by=user.uuid)
