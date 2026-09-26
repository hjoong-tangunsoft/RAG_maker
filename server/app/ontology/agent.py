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
    "You are a customer success assistant for Tangunsoft, a reseller of developer "
    "tools like JetBrains and GitHub Enterprise. You have access to the company's "
    "business ontology through tools: customers, contracts, licenses, support "
    "tickets, and vendors are all queryable. When the user asks about renewals, "
    "at-risk customers, or response plans, USE THE TOOLS - do not guess. "
    "LANGUAGE POLICY (STRICT, NON-NEGOTIABLE): "
    "1) Detect the user's language from their message. "
    "2) Your ENTIRE final answer MUST be in that single language. "
    "3) If the user writes in Korean (한국어), answer ONLY in Korean. Do NOT write "
    "any part of the answer in English or Chinese - not headings, not labels, "
    "not field names, nothing. Translate technical terms into natural Korean. "
    "4) NEVER repeat the same answer in another language. One answer, one language. "
    "5) This rule overrides any tendency to default to English. "
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
                await on_event({"type": "tool_call", "name": name, "arguments": args_json})
            result_json = tools.dispatch(name, args_json)
            if on_event:
                await on_event({
                    "type": "tool_result",
                    "name": name,
                    "summary": _summarize_result(result_json),
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
