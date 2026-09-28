"""Intent-router endpoint (Issue #37, P5 v3).

Step 1 baseline: `INTENT_ROUTER_ENABLED=false` → LiteLLM plain-proxy.
Step 2: protocol bypass rules (mellum, tools present, tool_calls history,
allowlist).
Step 3: B2 LLM classifier decides 'plain' | 'rag' | 'ontology'.
Step 4: dispatch to extracted service functions (no HTTP self-call).
Step 5: observability (router_request_id, latency, LiteLLM metadata,
shadow mode).

Rollout gate: env `INTENT_ROUTER_ENABLED` and `INTENT_ROUTER_SHADOW`.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .. import llm
from ..config import settings
from ..schemas import ChatCompletionRequest
from ..services.ontology_chat import ontology_chat_service
from ..services.rag_chat import rag_chat_service
from .intent import classify_intent

log = logging.getLogger("router")

router = APIRouter(prefix="/router/v1", tags=["intent-router"])

_HEADER_REQUEST_ID = "X-Router-Request-Id"


def _parse_csv_setting(csv: str) -> list[str]:
    """Parse comma-separated env-style string into a clean list of tokens."""
    return [x.strip() for x in csv.split(",") if x.strip()]


def _should_bypass(body: ChatCompletionRequest) -> tuple[bool, str]:
    """Decide whether to skip the classifier and plain-proxy to LiteLLM.

    Bypass rules (any match → skip classifier). Returns (True, reason)
    on first match, else (False, "").

    Rules:
    1. Model is in `INTENT_ROUTER_BYPASS_MODELS` (e.g. autocomplete).
    2. Message history contains a tool response or an assistant tool_call
       (multi-turn tool-calling conversation - client is mid-orchestration
       and the router must not interrupt).
    3. Model is not in `INTENT_ROUTER_ALLOWED_MODELS` (only allowlisted
       virtual models route through the classifier).

    HISTORICAL NOTE: An earlier version had a "tools_in_request" bypass
    (Rule 2 in the original handoff §8). It was removed because Continue.dev
    Agent mode sends 12 tools in EVERY request, so that rule fired 100% of
    the time and the classifier never ran. Downstream saw filesystem tools
    like `ls` / `read_file` and answered "단군소프트 매출 알려줘" with a
    naive `ls` call. See the router.dispatch() docstring for how we strip
    tools on rag/ontology routes so RAG can inject and Ontology can use
    its own tools.
    """
    model = (body.model or "").strip()

    bypass_models = _parse_csv_setting(settings.intent_router_bypass_models)
    if model in bypass_models:
        return True, f"model={model} in bypass_models"

    for m in body.messages:
        if m.role == "tool":
            return True, "tool_role_in_history"
        if m.tool_calls:
            return True, "tool_calls_in_history"

    allowed_models = _parse_csv_setting(settings.intent_router_allowed_models)
    if allowed_models and model not in allowed_models:
        return True, f"model={model} not in allowed_models"

    return False, ""


def _resolve_model(client_model: str | None) -> str:
    """Translate virtual model aliases to real backend models.

    Only allowlisted virtual models (INTENT_ROUTER_ALLOWED_MODELS) get
    resolved to `settings.default_model`. Real models (like `mellum-4b`,
    `qwen2.5-7b`) pass through unchanged so LiteLLM sees them verbatim.

    Without this, forwarding `qwen2.5-auto` verbatim to LiteLLM fails
    with `ProxyModelNotFoundError` because that alias only exists in
    the router's own namespace.
    """
    if not client_model:
        return settings.default_model
    allowed = _parse_csv_setting(settings.intent_router_allowed_models)
    if client_model.strip() in allowed:
        return settings.default_model
    return client_model


def _attach_request_id(resp: Any, request_id: str) -> Any:
    """Attach X-Router-Request-Id header to a Starlette Response.

    Passes StreamingResponse through unchanged too (it's a Response
    subclass so `.headers` works). Returns other objects (like raw
    dicts from unexpected paths) unchanged.
    """
    if isinstance(resp, Response):
        resp.headers[_HEADER_REQUEST_ID] = request_id
    return resp


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
) -> Any:
    """Intent-router entrypoint.

    Flow:
    1. Disabled → plain-proxy (Step 1 baseline).
    2. Bypass rules match → plain-proxy with logged reason (Step 2).
    3. Classify intent via B2 LLM tool-calling (Step 3).
    4. Shadow mode → log the decision but return plain-proxy (Step 5).
    5. Dispatch to the extracted service function (Step 4).

    Every branch generates a router_request_id (UUID4), measures
    latency, and threads the id into LiteLLM metadata on the plain-proxy
    path for LiteLLM_SpendLogs join.
    """
    request_id = uuid.uuid4().hex
    client_model = (body.model or "").strip() or "<unset>"
    total_start = time.perf_counter()

    # NOTE: We deliberately do NOT resolve body.model here. Bypass rules
    # (Rule 4 in particular) and the allowlist check must operate on the
    # ORIGINAL virtual model name (e.g. `qwen2.5-auto`). Only after we
    # decide the route do we translate to a real backend model, right
    # before the downstream call.

    if not settings.intent_router_enabled:
        body.model = _resolve_model(body.model)
        resp = await _proxy_to_litellm(body, request_id, route="plain")
        _log_summary(
            request_id=request_id,
            client_model=client_model,
            route="plain",
            bypass_reason="router_disabled",
            classifier_ms=0,
            total_start=total_start,
        )
        return _attach_request_id(resp, request_id)

    bypass, reason = _should_bypass(body)
    if bypass:
        body.model = _resolve_model(body.model)
        resp = await _proxy_to_litellm(body, request_id, route="plain")
        _log_summary(
            request_id=request_id,
            client_model=client_model,
            route="plain",
            bypass_reason=reason,
            classifier_ms=0,
            total_start=total_start,
        )
        return _attach_request_id(resp, request_id)

    classifier_start = time.perf_counter()
    route = await classify_intent(body)
    classifier_ms = int((time.perf_counter() - classifier_start) * 1000)

    # Now that the classifier has decided, translate the virtual alias
    # to a real backend model exactly once for the downstream call.
    resolved = _resolve_model(body.model)
    if resolved != body.model:
        log.info(
            "router model resolve: %s -> %s",
            body.model, resolved,
            extra={"router_request_id": request_id},
        )
        body.model = resolved

    if settings.intent_router_shadow:
        log.info(
            "router shadow: classified route=%s, returning plain-proxy",
            route,
            extra={"router_request_id": request_id, "shadow_route": route},
        )
        resp = await _proxy_to_litellm(
            body, request_id, route="plain", shadow_route=route,
        )
        _log_summary(
            request_id=request_id,
            client_model=client_model,
            route="plain",
            bypass_reason=f"shadow(would_route={route})",
            classifier_ms=classifier_ms,
            total_start=total_start,
        )
        return _attach_request_id(resp, request_id)

    resp = await _dispatch(route, body, request)
    _log_summary(
        request_id=request_id,
        client_model=client_model,
        route=route,
        bypass_reason="",
        classifier_ms=classifier_ms,
        total_start=total_start,
    )
    return _attach_request_id(resp, request_id)


async def _dispatch(
    route: str,
    body: ChatCompletionRequest,
    request: Request,
) -> Any:
    """Dispatch to the extracted service function for the given route.

    Unknown routes plain-proxy (defensive fallback). Service functions
    are called as regular Python awaits - no internal HTTP self-call.

    Tool handling per route:
    - `plain`: client tools kept (Continue.dev Agent mode uses them for
      coding tasks).
    - `rag`: client tools DROPPED so RAG service's `should_inject_rag`
      becomes True. Continue.dev's filesystem tools cannot answer
      company-knowledge questions - the retrieval answer replaces them.
    - `ontology`: client tools DROPPED so the ontology agent runs its
      own tools (`list_customers`, `list_customer_contracts`, ...)
      instead of being confused by Continue.dev's filesystem tools.

    NOTE: LiteLLM metadata (router_request_id) is only injected on the
    plain-proxy path today. RAG/Ontology routes still get log-based
    correlation via router_request_id but no LiteLLM_SpendLogs metadata.
    Threading metadata into those services is a future improvement.
    """
    if route == "rag":
        body.tools = None
        body.tool_choice = None
        return await rag_chat_service(body, request)
    if route == "ontology":
        body.tools = None
        body.tool_choice = None
        return await ontology_chat_service(body)
    return await _proxy_to_litellm(body, request_id=None, route="plain")


async def _proxy_to_litellm(
    body: ChatCompletionRequest,
    request_id: str | None,
    route: str,
    shadow_route: str | None = None,
) -> Any:
    """Direct passthrough to LiteLLM. Used when router is off/bypassed/shadow.

    Preserves the whole ChatCompletionRequest (including tools/tool_choice)
    so the caller sees identical behavior to /v1/chat/completions.

    When `request_id` is provided, injects it as LiteLLM metadata so
    LiteLLM_SpendLogs rows carry the correlation id. `shadow_route`
    (if set) records what the classifier WOULD have chosen.
    """
    messages = [m.model_dump(exclude_none=True) for m in body.messages]

    extra: dict[str, Any] | None = None
    if request_id:
        metadata: dict[str, Any] = {
            "router_request_id": request_id,
            "route": route,
        }
        if shadow_route is not None:
            metadata["shadow_route"] = shadow_route
        extra = {"metadata": metadata}

    if body.stream:
        async def gen():
            async for chunk in llm.stream_chat(
                messages=messages,
                model=body.model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                tools=body.tools,
                tool_choice=body.tool_choice,
                extra=extra,
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
        extra=extra,
    )
    return JSONResponse(resp)


def _log_summary(
    *,
    request_id: str,
    client_model: str,
    route: str,
    bypass_reason: str,
    classifier_ms: int,
    total_start: float,
) -> None:
    """Emit the per-request summary log line with all observability fields."""
    total_ms = int((time.perf_counter() - total_start) * 1000)
    log.info(
        "router rid=%s client_model=%s route=%s bypass=%s "
        "classifier_ms=%d total_ms=%d",
        request_id,
        client_model,
        route,
        bypass_reason or "-",
        classifier_ms,
        total_ms,
        extra={
            "router_request_id": request_id,
            "client_model": client_model,
            "route": route,
            "bypass_reason": bypass_reason,
            "classifier_latency_ms": classifier_ms,
            "total_latency_ms": total_ms,
        },
    )
