"""Intent-router endpoint (Issue #37, P5 v3).

Step 1 baseline: `INTENT_ROUTER_ENABLED=false` → LiteLLM plain-proxy.
Step 2: protocol bypass rules (mellum, tools present, tool_calls history,
allowlist).
Step 3: B2 LLM classifier decides 'plain' | 'rag' | 'ontology'.

Dispatch (Step 4) and observability (Step 5) still pending.

Rollout gate: env `INTENT_ROUTER_ENABLED` and `INTENT_ROUTER_SHADOW`.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import llm
from ..config import settings
from ..schemas import ChatCompletionRequest
from .intent import classify_intent

log = logging.getLogger("router")

router = APIRouter(prefix="/router/v1", tags=["intent-router"])


def _parse_csv_setting(csv: str) -> list[str]:
    """Parse comma-separated env-style string into a clean list of tokens."""
    return [x.strip() for x in csv.split(",") if x.strip()]


def _should_bypass(body: ChatCompletionRequest) -> tuple[bool, str]:
    """Decide whether to skip the classifier and plain-proxy to LiteLLM.

    Bypass rules (any match → skip classifier). Returns (True, reason)
    on first match, else (False, "").

    Rules:
    1. Model is in `INTENT_ROUTER_BYPASS_MODELS` (e.g. autocomplete).
    2. Request has `tools` array (client is orchestrating tool_calls).
    3. Message history contains a tool response or an assistant tool_call
       (multi-turn tool-calling conversation).
    4. Model is not in `INTENT_ROUTER_ALLOWED_MODELS` (only allowlisted
       virtual models route through the classifier).
    """
    model = (body.model or "").strip()

    bypass_models = _parse_csv_setting(settings.intent_router_bypass_models)
    if model in bypass_models:
        return True, f"model={model} in bypass_models"

    if body.tools:
        return True, "tools_in_request"

    for m in body.messages:
        if m.role == "tool":
            return True, "tool_role_in_history"
        if m.tool_calls:
            return True, "tool_calls_in_history"

    allowed_models = _parse_csv_setting(settings.intent_router_allowed_models)
    if allowed_models and model not in allowed_models:
        return True, f"model={model} not in allowed_models"

    return False, ""


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
) -> Any:
    """Intent-router entrypoint.

    Step 1: `INTENT_ROUTER_ENABLED=false` (default) → LiteLLM plain-proxy.
    Step 2: bypass rules match → plain-proxy with logged reason.

    Later steps (3-5) plug in:
    - B2 LLM classifier
    - Dispatch to plain/rag/ontology services
    - `router_request_id` observability + LiteLLM metadata
    """
    if not settings.intent_router_enabled:
        return await _proxy_to_litellm(body)

    bypass, reason = _should_bypass(body)
    if bypass:
        log.info("router bypass: %s", reason)
        return await _proxy_to_litellm(body)

    route = await classify_intent(body)
    log.info("router classified: route=%s", route)

    # TODO Step 4: dispatch to plain/rag/ontology
    # TODO Step 5: observability
    return await _proxy_to_litellm(body)


async def _proxy_to_litellm(body: ChatCompletionRequest) -> Any:
    """Direct passthrough to LiteLLM. Used when router is off/bypassed.

    Preserves the whole ChatCompletionRequest (including tools/tool_choice)
    so the caller sees identical behavior to /v1/chat/completions.
    """
    messages = [m.model_dump(exclude_none=True) for m in body.messages]

    if body.stream:
        async def gen():
            async for chunk in llm.stream_chat(
                messages=messages,
                model=body.model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                tools=body.tools,
                tool_choice=body.tool_choice,
            ):
                yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")

    resp = await llm.chat(
        messages=messages,
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        tools=body.tools,
        tool_choice=body.tool_choice,
    )
    return JSONResponse(resp)
