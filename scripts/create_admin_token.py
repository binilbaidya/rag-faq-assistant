"""Mint a short-lived ADMIN access token for the /v1/admin endpoints, signed with JWT_SECRET_KEY."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

import jwt

from app.auth import SUPERADMIN_ROLE
from app.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub", default="admin", help="User id stored in the token")
    parser.add_argument("--hours", type=float, default=1)
    args = parser.parse_args()

    secret = get_settings().jwt_secret_key
    if not secret:
        raise SystemExit("Set JWT_SECRET_KEY in .env first.")
    payload = {
        "sub": args.sub,
        "role": [SUPERADMIN_ROLE],
        "token_type": "access",
        "exp": datetime.now(UTC) + timedelta(hours=args.hours),
    }
    print(jwt.encode(payload, secret, algorithm="HS256"))


if __name__ == "__main__":
    main()
