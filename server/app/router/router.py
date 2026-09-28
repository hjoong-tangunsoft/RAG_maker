"""Intent-router endpoint (Issue #37, P5 v3).

Step 1 baseline: `INTENT_ROUTER_ENABLED=false` → LiteLLM plain-proxy.
Full dispatch (bypass rules, B2 classifier, service dispatch, observability)
ships in Steps 2-5.

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

log = logging.getLogger("router")

router = APIRouter(prefix="/router/v1", tags=["intent-router"])


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
) -> Any:
    """Intent-router entrypoint.

    Step 1: `INTENT_ROUTER_ENABLED=false` (default) → LiteLLM plain-proxy.
    Placeholder returns identical to a bare /v1/chat/completions call.

    Later steps (2-5) plug in:
    - Protocol bypass (mellum, tools present, tool_calls history, allowlist)
    - B2 LLM classifier
    - Dispatch to plain/rag/ontology services
    - `router_request_id` observability + LiteLLM metadata
    """
    if not settings.intent_router_enabled:
        return await _proxy_to_litellm(body)

    # TODO Steps 2-5: bypass rules → classifier → dispatch → observability
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
