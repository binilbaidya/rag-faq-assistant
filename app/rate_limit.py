from __future__ import annotations

import ipaddress
import logging
import time
from collections import deque
from threading import Lock

from fastapi import HTTPException, Request, status

from app.auth import _bearer_token
from app.config import get_settings


logger = logging.getLogger(__name__)


class SlidingWindowLimiter:
    """Per-process fixed-capacity sliding window.

    Deliberately in-memory: with N workers the effective ceiling is N x the configured rate, which
    is fine for shielding LLM spend but is not a security control. Move to Redis if it ever needs
    to be exact across workers.
    """

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def check(self, key: str) -> float | None:
        """Returns None when allowed, else the seconds to wait before retrying."""
        if self.limit <= 0:
            return None
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(0.0, hits[0] + self.window_seconds - now)
            hits.append(now)
            if len(self._hits) > 10_000:
                self._prune(cutoff)
        return None

    def _prune(self, cutoff: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            self._hits.pop(key, None)


_settings = get_settings()
_chat_limiter = SlidingWindowLimiter(_settings.chat_rate_limit, _settings.chat_rate_window_seconds)
_daily_limiter = SlidingWindowLimiter(_settings.chat_daily_limit, 24 * 60 * 60)


def enforce_client_rate_limit(request: Request) -> None:
    _enforce(_chat_limiter.check(_client_key(request)), "Too many chat requests. Please wait a moment and try again.")


def enforce_chat_rate_limit(request: Request) -> None:
    enforce_client_rate_limit(request)
    _enforce(_daily_limiter.check("all"), "The assistant has reached today's question limit. Please try again tomorrow.")


def _enforce(retry_after: float | None, detail: str) -> None:
    if retry_after is None:
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=detail,
        headers={"Retry-After": str(max(1, int(retry_after) + 1))},
    )


def _client_key(request: Request) -> str:
    token = _bearer_token(request.headers.get("authorization"))
    if token:
        return f"t:{token[-32:]}"
    return f"ip:{client_ip(request) or 'unknown'}"


def client_ip(request: Request) -> str | None:
    """The visitor's IP, read the way CLIENT_IP_HEADER says this deployment's proxies pass it on."""
    direct = request.client.host if request.client else None
    header = get_settings().client_ip_header
    if header == "none":
        # No proxy in front: any forwarding header came from the client itself, so ignore them.
        return direct
    if header != "x-forwarded-for":
        # A CDN's own header (e.g. CF-Connecting-IP) holds the single IP the CDN saw; it overwrites any client value.
        return request.headers.get(header, "").split(",")[0].strip() or direct
    # Proxies append to X-Forwarded-For, so the left end is client-supplied. Walking from the right,
    # the host's own private hops (e.g. a 10.x load balancer) are skipped; the first public IP is the visitor.
    forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
    for candidate in reversed(forwarded):
        try:
            if ipaddress.ip_address(candidate).is_global:
                return candidate
        except ValueError:
            continue
    if forwarded:
        logger.warning("No public IP in X-Forwarded-For: %s", forwarded)
    return direct
