"""B2 LLM tool-calling intent classifier (Issue #37 Step 3).

Given a ChatCompletionRequest, decide which route the user's last message
belongs to: 'plain' | 'rag' | 'ontology'.

Uses the same Qwen backend via LiteLLM with a fixed classifier system
prompt and 3 route tools. `tool_choice="required"` forces the model to
emit exactly one tool_call (no free-form text). On any error or
ambiguity, falls back to `INTENT_ROUTER_DEFAULT`.

The classifier deliberately calls `settings.default_model` (the real
backend, e.g. qwen2.5-7b) not the virtual alias that reaches the router,
to avoid infinite recursion.
"""
from __future__ import annotations

import logging
from typing import Any

from .. import llm
from ..config import settings
from ..schemas import ChatCompletionRequest

log = logging.getLogger("router.intent")

_VALID_ROUTES = ("plain", "rag", "ontology")

# Three no-arg route tools. tool_choice="required" forces exactly one call.
INTENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "route_plain",
            "description": (
                "일반 대화, 인사, 상식 질문, 코드 작성/편집/설명 요청 등 "
                "회사 내부 문서나 관계형 DB 지식이 필요 없는 모든 경우."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_rag",
            "description": (
                "회사 내부 문서, 매뉴얼, 정책, FAQ, 프로덕트 기능 설명 "
                "등 사내 지식 베이스 검색이 필요한 질문."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_ontology",
            "description": (
                "고객 정보, 계약, 라이선스, 특정 사용자 데이터 등 관계형 "
                "DB 조회가 필요한 질문. 예: '홍길동 고객의 계약 목록', "
                "'이 고객이 보유한 라이선스', 'X 회사의 계약 현황'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

_CLASSIFIER_SYSTEM_PROMPT = """당신은 사용자 질문의 라우팅 결정만 수행하는 분류기입니다.

세 가지 라우팅 옵션 중 정확히 하나를 tool_call 로 호출해야 합니다:
- route_plain: 일반 대화 / 코드 작성 / 상식 질문 (회사 지식 불필요)
- route_rag: 회사 내부 문서·매뉴얼·정책 검색이 필요한 질문
- route_ontology: 고객·계약·라이선스 등 관계형 DB 조회가 필요한 질문

절대 텍스트로 답변하지 마세요. 반드시 tool_call 만 반환하세요."""


def _last_user_content(body: ChatCompletionRequest) -> str | None:
    """Return the content of the most recent user-role message, or None."""
    for m in reversed(body.messages):
        if m.role == "user" and m.content:
            return m.content
    return None


def _extract_route_from_response(resp: dict[str, Any], default: str) -> str:
    """Pure helper: pull the route label out of an LLM tool-call response.

    Returns one of `_VALID_ROUTES`. Falls back to `default` on any
    malformed structure or unknown tool name.
    """
    try:
        choices = resp.get("choices") or []
        if not choices:
            return default
        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return default
        fn_name = (tool_calls[0].get("function") or {}).get("name") or ""
        if not fn_name.startswith("route_"):
            return default
        route = fn_name[len("route_"):]
        if route in _VALID_ROUTES:
            return route
        return default
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        log.warning("malformed classifier response: %s", e)
        return default


async def classify_intent(body: ChatCompletionRequest) -> str:
    """Classify user intent → 'plain' | 'rag' | 'ontology'.

    Fallback to `INTENT_ROUTER_DEFAULT` on:
    - No user message in history
    - Classifier LLM error
    - Malformed tool_call response
    - Unknown route name

    Notes:
    - Uses `settings.default_model` (the real backend) to avoid
      recursively routing through the virtual alias.
    - Only the last user message is passed. Multi-turn context is a
      future extension.
    """
    default = settings.intent_router_default

    last_user = _last_user_content(body)
    if not last_user:
        log.info("classifier: no user content, fallback=%s", default)
        return default

    classifier_messages = [
        {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": last_user},
    ]

    try:
        resp = await llm.chat(
            messages=classifier_messages,
            model=settings.default_model,
            temperature=0.0,
            max_tokens=20,
            tools=INTENT_TOOLS,
            tool_choice="required",
        )
    except Exception as e:  # httpx errors, timeouts, upstream 5xx
        log.warning("classifier LLM call failed: %s", e)
        return default

    route = _extract_route_from_response(resp, default)
    log.info("classifier: route=%s", route)
    return route
