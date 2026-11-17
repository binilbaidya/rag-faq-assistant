from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.admin_ai import router as admin_ai_router
from app.api.routes.chat import router as chat_router
from app.api.routes.faqs import router as faqs_router
from app.api.routes.health import router as health_router
from app.config import get_settings


logger = logging.getLogger(__name__)

DOCS_ENVIRONMENTS = {"development", "local", "dev"}


def _configure_logging(settings) -> None:
    """Uvicorn configures only its own loggers; without this `app.*` records have no handler."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        force=True,
    )


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings)
    docs_enabled = settings.environment.strip().lower() in DOCS_ENVIRONMENTS
    application = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    if not docs_enabled:
        logger.info("API docs disabled for ENVIRONMENT=%s", settings.environment)
    _configure_cors(application, settings)
    application.include_router(health_router)
    application.include_router(admin_ai_router)
    application.include_router(faqs_router)
    application.include_router(chat_router)
    return application


def _configure_cors(application: FastAPI, settings) -> None:
    if settings.cors_allow_all:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        return

    if not settings.cors_allowed_origins:
        logger.warning(
            "No CORS middleware registered: set CORS_ALLOWED_ORIGINS (or CORS_ALLOW_ALL). "
            "Browser requests will fail until one is configured."
        )
        return

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )


app = create_app()
