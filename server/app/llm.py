"""Async client for the local LiteLLM proxy (OpenAI-compatible)."""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

from .config import settings

log = logging.getLogger(__name__)

_CHAT_URL = f"{settings.litellm_url}/v1/chat/completions"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.litellm_api_key}",
        "Content-Type": "application/json",
    }


async def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Non-streaming chat completion. Returns the full JSON response."""
    payload: dict[str, Any] = {
        "model": model or settings.default_model,
        "messages": messages,
        "temperature": settings.rag_temperature if temperature is None else temperature,
        "max_tokens": settings.rag_max_tokens if max_tokens is None else max_tokens,
        "stream": False,
    }
    if extra:
        payload.update(extra)
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(_CHAT_URL, json=payload, headers=_headers())
        r.raise_for_status()
        return r.json()


async def stream_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
) -> AsyncIterator[bytes]:
    """Stream SSE bytes from the upstream chat endpoint straight to the caller."""
    payload: dict[str, Any] = {
        "model": model or settings.default_model,
        "messages": messages,
        "temperature": settings.rag_temperature if temperature is None else temperature,
        "max_tokens": settings.rag_max_tokens if max_tokens is None else max_tokens,
        "stream": True,
    }
    if extra:
        payload.update(extra)
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", _CHAT_URL, json=payload, headers=_headers()
        ) as r:
            r.raise_for_status()
            async for chunk in r.aiter_raw():
                if chunk:
                    yield chunk


async def list_models() -> dict[str, Any]:
    """Proxy list of upstream models."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{settings.litellm_url}/v1/models", headers=_headers())
        r.raise_for_status()
        return r.json()


async def health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{settings.litellm_url}/health/liveliness")
            return r.status_code < 500
    except Exception:  # noqa: BLE001
        return False
