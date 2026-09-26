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
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Non-streaming chat completion. Returns the full JSON response.

    When `tools` is provided, they are passed to the upstream LLM verbatim
    (OpenAI tool-calling spec). Tool responses come back as
    choices[0].message.tool_calls per the spec.
    """
    payload: dict[str, Any] = {
        "model": model or settings.default_model,
        "messages": messages,
        "temperature": settings.rag_temperature if temperature is None else temperature,
        "max_tokens": settings.rag_max_tokens if max_tokens is None else max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    if extra:
        payload.update(extra)
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(_CHAT_URL, json=payload, headers=_headers())
        r.raise_for_status()
        return r.json()


async def stream_chat(
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> AsyncIterator[bytes]:
    """Stream SSE bytes from the upstream chat endpoint straight to the caller.

    Tool-calling: streams `delta.tool_calls` chunks per OpenAI spec when
    `tools` are provided. Callers passing tools should use this path
    directly (bypassing the hanja-guard buffer) because tool_calls are
    structured JSON not natural text.
    """
    payload: dict[str, Any] = {
        "model": model or settings.default_model,
        "messages": messages,
        "temperature": settings.rag_temperature if temperature is None else temperature,
        "max_tokens": settings.rag_max_tokens if max_tokens is None else max_tokens,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
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
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Zero-tolerance hanja guard (Phase E).

    Up to (1 + max_retries) attempts to generate a response with
    hanja_count() <= settings.hanja_threshold. Each retry uses a reinforced
    system prompt and a lower temperature. If all attempts fail, apply
    _convert_hanja_to_hangul() to the last response as a last-resort
    transliteration (李贤中 → 이현중).

    Tool-call responses (content=None with tool_calls) skip the hanja
    check because tool_calls are structured JSON, not natural language.

    Non-streaming callers use this directly. Streaming callers go through
    the buffered-SSE wrapper in main.py so they get the same guarantees.
    """
    current_messages = messages
    base_temp = settings.rag_temperature if temperature is None else temperature
    # Escalating strictness: temperature drops with each retry attempt.
    temps = [base_temp, min(base_temp / 2, 0.05), 0.02]

    resp: dict[str, Any] = {}
    text = ""
    n = 0
    for attempt in range(max_retries + 1):
        temp_this = temps[attempt] if attempt < len(temps) else 0.02
        resp = await chat(
            current_messages, model=model, temperature=temp_this,
            max_tokens=max_tokens, tools=tools, tool_choice=tool_choice,
            extra=extra,
        )
        try:
            message = resp["choices"][0]["message"]
            text = message.get("content")
        except (KeyError, IndexError, TypeError):
            return resp  # unexpected shape - can't guard

        # Skip hanja check on tool_call responses (structured JSON, not text)
        if text is None:
            return resp

        n = hanja_count(text)
        if n <= settings.hanja_threshold:
            if attempt > 0:
                log.info("hanja resolved after retry %d (final=%d)", attempt, n)
            return resp

        if attempt < max_retries:
            next_temp = temps[attempt + 1] if attempt + 1 < len(temps) else 0.02
            log.warning(
                "attempt %d: hanja=%d > threshold=%d, retrying (temp=%.3f)",
                attempt, n, settings.hanja_threshold, next_temp,
            )
            current_messages = _reinforce_korean_only(current_messages)
        else:
            log.error(
                "attempt %d: hanja=%d still present after all retries, "
                "applying last-resort conversion",
                attempt, n,
            )

    # All retries exhausted - convert hanja to hangul as last resort
    converted = _convert_hanja_to_hangul(text)
    final_n = hanja_count(converted)
    log.warning(
        "last-resort hanja conversion: %d -> %d (%s)",
        n, final_n, "clean" if final_n == 0 else "partial",
    )
    resp["choices"][0]["message"]["content"] = converted
    return resp


def _convert_hanja_to_hangul(text: str) -> str:
    """Last-resort: transliterate hanja to Korean phonetic reading.

    Uses the `hanja` PyPI package for K-reading conversion (e.g.
    李贤中 → 이현중, 業務 → 업무). Falls back to stripping hanja chars if
    the library is not installed or raises.

    Note: transliteration is character-level; whole-sentence translation
    (e.g. 应该完成 → 해야 할 일) is out of scope. This is a safety net,
    not a translator.
    """
    try:
        import hanja  # noqa: PLC0415 - optional dep, lazy import
        return hanja.translate(text, "substitution")
    except ImportError:
        log.warning("hanja lib not installed; stripping hanja chars instead")
        return _HANJA_RE.sub("", text)
    except Exception as e:  # noqa: BLE001
        log.error("hanja conversion failed (%s); stripping instead", e)
        return _HANJA_RE.sub("", text)
