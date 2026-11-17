from __future__ import annotations

from typing import Annotated, Any

import jwt
from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings


SUPERADMIN_ROLE = "ADMIN"


class AuthenticatedUser(BaseModel):
    uuid: str
    roles: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_superadmin(self) -> bool:
        return SUPERADMIN_ROLE in self.roles


def resolve_user_from_authorization(authorization: str | None) -> AuthenticatedUser:
    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization token is missing.")

    secret = get_settings().jwt_secret_key
    if not secret:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="JWT secret is not configured.")

    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization token.") from exc

    if payload.get("token_type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh tokens cannot be used here.")

    user_uuid = str(payload.get("sub") or "").strip()
    if not user_uuid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization token has no user.")

    roles = payload.get("role", [])
    if isinstance(roles, str):
        roles = [roles]
    elif not isinstance(roles, list):
        roles = []

    return AuthenticatedUser(uuid=user_uuid, roles=[str(role) for role in roles], payload=payload)


def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
) -> AuthenticatedUser:
    return resolve_user_from_authorization(authorization)


def get_optional_user(
    authorization: Annotated[str | None, Header()] = None,
) -> AuthenticatedUser | None:
    if not _bearer_token(authorization):
        return None
    return resolve_user_from_authorization(authorization)


def require_superadmin_user(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> AuthenticatedUser:
    if not user.is_superadmin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Superadmin access is required.")
    return user


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()
