"""Agent chat loop: LLM + tool calling over the ontology.

Flow:
    user message
      -> LLM (with TOOLS schemas)
      -> if tool_calls -> execute locally -> append tool results -> loop
      -> else -> final text answer

The LLM is unchanged (qwen2.5-7b). What changes is that we hand it a list of
verbs (tools) it can call. This is where "RAG (finding) + Ontology (understanding)
+ Agent (acting)" ladder culminates.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from .. import llm
from . import tools

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a customer success assistant for Tangunsoft, a reseller of "
    "developer tools (JetBrains, GitHub Enterprise). You have tools to query "
    "the business ontology: customers, contracts, licenses, support tickets, "
    "vendors.\n\n"
    "TOOL POLICY:\n"
    "- For questions about customers, contracts, renewals, licenses, tickets, "
    "or vendors, USE THE PROVIDED TOOLS. Do not guess.\n"
    "- For general questions (greetings, company definitions, unrelated topics, "
    "questions the tools cannot answer), answer DIRECTLY from your knowledge "
    "WITHOUT calling any tool. Do not fabricate tool names.\n"
    "- NEVER call a tool that is not in the provided tools list. NEVER mention "
    "tool names like 'search_web', 'read_file' that you do not actually have.\n\n"
    "LANGUAGE POLICY (STRICT, NON-NEGOTIABLE):\n"
    "1) Detect the user's language from their message.\n"
    "2) Your ENTIRE final answer MUST be in that single language.\n"
    "3) If the user writes in Korean, answer ONLY in Korean - no English "
    "sentences, no Chinese characters, no mixed language, no romanized "
    "Chinese pinyin.\n"
    "4) NEVER repeat the answer in another language.\n"
    "5) This rule overrides any default tendency to use English or Chinese.\n\n"
    "Be concise and factual."
)


def _has_hangul(text: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)


class AgentTrace(dict[str, Any]):
    """Structured trace of one agent run - useful for debugging + UI display."""


async def run(
    user_message: str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    max_iterations: int = 5,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> AgentTrace:
    """Run the agent loop until a final text answer or iteration limit.

    on_event: optional async callback invoked at each tool_call and tool_result
    so a streaming client (SSE wrapper) can show progress in real time.
    """
    base_system = system_prompt or DEFAULT_SYSTEM_PROMPT
    if _has_hangul(user_message):
        base_system += (
            "\n\nCONFIRMED USER LANGUAGE: Korean. "
            "Your final answer MUST be entirely in Korean. No English sentences."
        )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": base_system},
        {"role": "user", "content": user_message},
    ]
    tool_calls_log: list[dict[str, Any]] = []
    iterations = 0
    final_text = ""
    used_model = model

    while iterations < max_iterations:
        iterations += 1
        if on_event:
            await on_event({"type": "thinking", "iteration": iterations})
        resp = await llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra={"tools": tools.TOOLS, "tool_choice": "auto"},
        )
        used_model = resp.get("model", used_model)
        choice = resp["choices"][0]["message"]
        tc_list = choice.get("tool_calls") or []

        if not tc_list:
            final_text = choice.get("content") or ""
            break

        messages.append({
            "role": "assistant",
            "content": choice.get("content") or "",
            "tool_calls": tc_list,
        })
        for tc in tc_list:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args_json = fn.get("arguments", "") or ""
            log.info("agent tool_call: %s(%s)", name, args_json)
            if on_event:
                await on_event({
                    "type": "tool_call",
                    "name": name,
                    "arguments": args_json,
                    "narration": _narrate_call(name, args_json),
                })
            result_json = tools.dispatch(name, args_json)
            if on_event:
                await on_event({
                    "type": "tool_result",
                    "name": name,
                    "narration": _narrate_result(name, result_json),
                    "preview": result_json[:600],
                })
            tool_calls_log.append({
                "id": tc.get("id"),
                "name": name,
                "arguments": args_json,
                "result": result_json,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "name": name,
                "content": result_json,
            })
    else:
        final_text = (
            "[agent: max iterations reached without a final answer; "
            "last tool results are in the trace]"
        )

    return AgentTrace({
        "answer": final_text,
        "model": used_model,
        "iterations": iterations,
        "tool_calls": tool_calls_log,
    })


def _narrate_call(name: str, args_json: str) -> str:
    try:
        args = json.loads(args_json or "{}")
    except (ValueError, TypeError):
        return f"`{name}` 도구를 호출합니다."
    if name == "renewal_risk":
        parts: list[str] = []
        if v := args.get("vendor_name"):
            parts.append(f"벤더 **{v}**")
        if d := args.get("days_until_renewal"):
            parts.append(f"갱신일 {d}일 이내")
        if w := args.get("recent_window_days"):
            parts.append(f"최근 {w}일 창")
        if t := args.get("min_recent_tickets"):
            parts.append(f"지원티켓 {t}건 이상")
        crit = ", ".join(parts) if parts else "기본 조건"
        return f"`renewal_risk` 도구로 갱신 위험 고객을 조회합니다. 조건: {crit}."
    if name == "draft_response_plan":
        cid = args.get("customer_id", "?")
        return f"`draft_response_plan` 도구로 고객 **{cid}** 의 대응안 초안을 작성합니다."
    return f"`{name}` 도구를 호출합니다."


def _narrate_result(name: str, result_json: str) -> str:
    try:
        data = json.loads(result_json)
    except (ValueError, TypeError):
        return "결과를 받았습니다."
    if isinstance(data, dict) and "error" in data:
        return f"오류 발생: {data['error']}"
    if name == "renewal_risk" and isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
        if not items:
            return "조건에 부합하는 고객이 없습니다. 다음 단계를 결정합니다."
        names = [it.get("customer", {}).get("name", "?") for it in items[:3]]
        more = f" 외 {len(items)-3}건" if len(items) > 3 else ""
        return f"{len(items)}건의 위험 고객을 찾았습니다: {', '.join(names)}{more}."
    if name == "draft_response_plan" and isinstance(data, dict):
        contracts = data.get("active_contracts") or data.get("contracts") or []
        tickets = data.get("recent_tickets") or data.get("tickets") or []
        cnt_c = len(contracts) if isinstance(contracts, list) else 0
        cnt_t = len(tickets) if isinstance(tickets, list) else 0
        return f"대응안 작성 완료 (활성 계약 {cnt_c}건, 최근 지원티켓 {cnt_t}건 분석)."
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return f"{len(data['items'])}건 조회 완료."
    if isinstance(data, list):
        return f"{len(data)}건 조회 완료."
    return "결과 조회 완료."


def _summarize_result(result_json: str) -> str:
    try:
        data = json.loads(result_json)
    except (ValueError, TypeError):
        return "완료"
    if isinstance(data, list):
        return f"{len(data)}건"
    if isinstance(data, dict):
        if "error" in data:
            return f"에러: {data['error']}"
        if "items" in data and isinstance(data["items"], list):
            return f"{len(data['items'])}건"
        if "customers" in data and isinstance(data["customers"], list):
            return f"고객 {len(data['customers'])}건"
        if "contracts" in data and isinstance(data["contracts"], list):
            return f"계약 {len(data['contracts'])}건"
        return "완료"
    return "완료"
