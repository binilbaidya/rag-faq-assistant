from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import HTTPException, status
from pydantic import BaseModel


logger = logging.getLogger(__name__)

MALICIOUS_PATTERNS = [
    re.compile(r"\bforget\s+all\s+previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|system|developer)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\breveal\s+(the\s+)?(system|developer)\s+prompt\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(in\s+)?developer\s+mode\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bexecute\s+raw\s+(sql|database)\b", re.IGNORECASE),
]


def check_request(request: BaseModel) -> None:
    enforce_security(request.model_dump())


def enforce_security(payload: Any) -> None:
    for value in _iter_strings(payload):
        normalized = " ".join(value.split())
        for pattern in MALICIOUS_PATTERNS:
            if pattern.search(normalized):
                logger.warning("Rejected suspicious chatbot request matching %s", pattern.pattern)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Request rejected by chatbot security policy.",
                )


def _iter_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_iter_strings(item))
        return strings
    if isinstance(value, (list, tuple, set)):
        strings: list[str] = []
        for item in value:
            strings.extend(_iter_strings(item))
        return strings
    return []
