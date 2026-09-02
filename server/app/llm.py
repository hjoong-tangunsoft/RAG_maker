"""Async client for the local LiteLLM proxy (OpenAI-compatible)."""
from __future__ import annotations

import logging
import re
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


# ---- Korean Purity Guard (L3 - Detection + Retry) ----
#
# Qwen 2.5 heavily trained on Chinese sometimes leaks 한자 (CJK Unified
# Ideographs) into Korean answers. L1 (prompt) + L2 (temperature) block
# most cases; L3 catches the rest by regenerating with a reinforced
# prompt if the response exceeds `hanja_threshold`.
#
# Hangul (한글, 0xAC00-0xD7AF) is a separate unicode range and never
# triggers this check.

_HANJA_RE = re.compile(r"[\u4E00-\u9FFF]")


def hanja_count(text: str) -> int:
    """Count CJK Unified Ideographs in text (Chinese/Japanese chars, not Hangul)."""
    return len(_HANJA_RE.findall(text))


def _reinforce_korean_only(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Append Korean-only reinforcement to the system message for retry."""
    reinforcement = (
        "\n\n【중요】 이전 답변 시도에 한자가 포함되어 재시도합니다. "
        "이번에는 반드시 순수 한국어(한글)로만 답변하세요. "
        "한자·중국어 문자 절대 사용 금지. 한자어는 한글 발음으로 표기하세요. "
        "예: 業務->업무, 會社->회사, 資料->자료, 情報->정보"
    )
    new_messages: list[dict[str, str]] = []
    seen_system = False
    for m in messages:
        if m.get("role") == "system" and not seen_system:
            new_messages.append({
                "role": "system",
                "content": m.get("content", "") + reinforcement,
            })
            seen_system = True
        else:
            new_messages.append(m)
    if not seen_system:
        new_messages.insert(0, {"role": "system", "content": reinforcement.strip()})
    return new_messages


async def chat_guarded(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
    max_retries: int = 1,
) -> dict[str, Any]:
    """chat() + retry once if response contains too many CJK ideographs.

    Streaming callers should use stream_chat() directly (mid-stream retry
    is not possible).
    """
    resp = await chat(
        messages, model=model, temperature=temperature,
        max_tokens=max_tokens, extra=extra,
    )
    try:
        text = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return resp  # unexpected shape, don't crash

    n = hanja_count(text)
    if n <= settings.hanja_threshold or max_retries <= 0:
        return resp

    log.warning("hanja leak detected (%d chars), regenerating with reinforced prompt", n)
    retry_messages = _reinforce_korean_only(messages)
    resp2 = await chat(
        retry_messages,
        model=model,
        temperature=0.05,  # extra-low for strict retry
        max_tokens=max_tokens,
        extra=extra,
    )
    try:
        text2 = resp2["choices"][0]["message"]["content"]
        n2 = hanja_count(text2)
        if n2 > 0:
            log.error("hanja still present after retry: %d chars", n2)
        else:
            log.info("retry successful, hanja removed")
    except (KeyError, IndexError, TypeError):
        pass
    return resp2
