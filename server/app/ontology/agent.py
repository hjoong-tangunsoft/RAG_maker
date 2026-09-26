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
    "ENVIRONMENT: This is an internal air-gapped deployment. You have NO web "
    "access, NO file system access, NO shell. The ONLY tools you can call are "
    "the ones provided in the tools list. Do not invent tools. Do not suggest "
    "'searching the web', '웹 검색', 'search_web', 'read_file', 'run_command' "
    "or any tool not in the list.\n\n"
    "TOOL POLICY:\n"
    "- Ontology questions (customers, contracts, renewals, tickets, vendors) "
    "MUST use the ontology tools (renewal_risk, draft_response_plan).\n"
    "- Questions about JIRA issues, documents, notes, wiki pages, exported "
    "tickets, PDFs, or ANY factual content outside the customer/contract "
    "ontology MUST use `rag_search` FIRST. The RAG knowledge base already "
    "contains our JIRA export - never claim you need JIRA API/web access.\n"
    "- General questions (greetings, definitions, unrelated topics) answer "
    "DIRECTLY from your own knowledge. Do not call any tool.\n"
    "- If rag_search returns no relevant passages, say so honestly. Do not "
    "propose external actions.\n\n"
    "NEVER FABRICATE NUMBERS OR FACTS:\n"
    "- Do NOT state counts (customers, contracts, tickets, licenses) unless a "
    "tool response provided that exact number in this turn. If no tool covers "
    "the question, say '해당 정보는 현재 도구로 조회할 수 없습니다.' honestly.\n"
    "- Do NOT emit tool invocations as plain text or code "
    "(e.g. `renewal_risk(vendor_name=\"X\")` or `rag_search {\"query\": \"X\"}`). "
    "If you need a tool, use the tool_calls channel. Prose must be natural "
    "language for the user, never a code fragment.\n\n"
    "CONVERSATION CONTEXT (IMPORTANT):\n"
    "The user message you receive has ALREADY been rewritten to be "
    "self-contained using prior conversation context. Treat it as a "
    "complete standalone question and choose tools accordingly. If it "
    "refers to a vendor/customer/subject, that subject is the current "
    "target - use the appropriate tool with those parameters. Never skip "
    "a tool call because the topic feels familiar from earlier turns.\n\n"
    "LANGUAGE POLICY (STRICT, NON-NEGOTIABLE):\n"
    "1) Detect the user's language from their message.\n"
    "2) Your ENTIRE final answer MUST be in that single language.\n"
    "3) If the user writes in Korean, answer ONLY in Korean - no English "
    "sentences, no Chinese characters, no mixed language, no romanized "
    "Chinese pinyin, no Chinese instructions to yourself.\n"
    "4) NEVER repeat the answer in another language.\n"
    "5) This rule overrides any default tendency to use English or Chinese.\n\n"
    "Be concise and factual."
)


def _has_hangul(text: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)


def _has_chinese(text: str) -> bool:
    # CJK Unified Ideographs. Korean answers must not contain Han characters;
    # 7B model sometimes falls back to Chinese for unknown-vocabulary phrases.
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


_REWRITE_SYSTEM = (
    "You rewrite the user's latest message ONLY IF it cannot stand alone. "
    "Most messages are already self-contained and MUST be returned verbatim.\n\n"
    "Rules:\n"
    "1) Output ONLY the resulting question. No prefix, no quotes, no "
    "explanation.\n"
    "2) Return the latest message UNCHANGED when it already contains its own "
    "subject/verb and makes sense on its own. Examples that MUST pass through:\n"
    "   - '갱신해야될 고객사 알려줘' (has subject '고객사' + verb '알려줘')\n"
    "   - 'JetBrains 위험 고객 알려줘'\n"
    "   - 'JIRA에 어떤 글들 있어?'\n"
    "3) Rewrite ONLY when the latest message clearly depends on the previous "
    "turn - pronouns, ellipsis, or bare noun phrases with no verb. Examples:\n"
    "   - '깃허브는?' after a JetBrains question -> '깃허브 갱신 위험 고객은?'\n"
    "   - '그럼 LG는?' -> resolve LG in the same context\n"
    "   - '이유는?' -> 'why is that?' in the same context\n"
    "4) When the new message names a DIFFERENT subject (a different vendor, "
    "customer, or entity), REPLACE the prior subject entirely. NEVER "
    "concatenate the old and new subjects. Examples:\n"
    "   - prev: 'JetBrains 갱신 리스크', new: '단군소프트 말이야' -> "
    "'단군소프트 갱신 리스크는?' (NOT 'JetBrains 단군소프트 갱신...')\n"
    "   - prev: 'Adobe 위험 고객은?', new: 'MongoDB는?' -> "
    "'MongoDB 위험 고객은?' (NOT 'Adobe MongoDB 위험...')\n"
    "5) NEVER inject a subject (vendor/name/entity) that is not present in "
    "the latest message OR the immediately prior turn. If the latest message "
    "says 'all customers', do not narrow it to one vendor from earlier turns.\n"
    "6) Preserve the user's original language.\n"
    "7) When in doubt, return unchanged."
)


async def _rewrite_query(history: list[dict[str, Any]], last_user: str) -> str:
    """Rewrite an ambiguous follow-up into a self-contained question.

    Only the immediately preceding 1-2 turns are used as context so a fresh
    complete question later in the conversation does not get polluted by
    stale subjects (e.g. GitHub from 10 turns ago).
    """
    prior = history[:-1]
    if not prior or not last_user.strip():
        return last_user
    recent = prior[-2:]
    convo_lines = [
        f"{m.get('role','?')}: {(m.get('content') or '').strip()}"
        for m in recent
        if m.get("content")
    ]
    if not convo_lines:
        return last_user
    prompt = (
        "Prior conversation (most recent turns only):\n"
        + "\n".join(convo_lines)
        + f"\n\nLatest user message: {last_user}\n\nRewritten (or unchanged) question:"
    )
    try:
        resp = await llm.chat(
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        text = (resp["choices"][0]["message"].get("content") or "").strip()
        # strip common wrappers a small model likes to add
        for prefix in ("Rewritten:", "rewritten:", "Question:", "질문:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        text = text.strip('"').strip("'").strip()
        if not text:
            return last_user
        # Defense: small models sometimes echo a prior assistant refusal instead
        # of rewriting. Reject rewrites that (a) balloon in length AND (b) look
        # like refusal / meta commentary rather than a question.
        refusal_markers = (
            "도구가 없", "찾을 수 없", "다른 방법", "제공되지 않", "추가 지침",
            "sorry", "cannot", "unable to", "i don't have",
        )
        low = text.lower()
        looks_like_refusal = any(m in low for m in refusal_markers)
        much_longer = len(text) > max(80, len(last_user) * 3)
        if looks_like_refusal or much_longer:
            log.warning(
                "query rewrite rejected (refusal=%s longer=%s): %r",
                looks_like_refusal, much_longer, text[:120],
            )
            return last_user
        return text
    except Exception as e:  # noqa: BLE001
        log.warning("query rewrite failed, falling back to raw: %s", e)
        return last_user


class AgentTrace(dict[str, Any]):
    """Structured trace of one agent run - useful for debugging + UI display."""


async def run(
    conversation: list[dict[str, Any]] | str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    max_iterations: int = 5,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> AgentTrace:
    """Run the agent loop until a final text answer or iteration limit.

    conversation: full message list ([{role, content}, ...]) OR a single user
    string (legacy). Multi-turn history is required for follow-ups like
    '깃허브는?' to resolve against the prior turn.

    on_event: optional async callback invoked at each tool_call and tool_result
    so a streaming client (SSE wrapper) can show progress in real time.
    """
    # normalize: legacy str -> single-user list
    if isinstance(conversation, str):
        history: list[dict[str, Any]] = [{"role": "user", "content": conversation}]
    else:
        # strip any pre-existing system messages; we own the system prompt
        history = [m for m in conversation if m.get("role") != "system"]
    last_user = next(
        (m.get("content", "") for m in reversed(history) if m.get("role") == "user"),
        "",
    )

    rewritten = await _rewrite_query(history, last_user)
    if rewritten != last_user and on_event:
        await on_event({"type": "rewrite", "original": last_user, "rewritten": rewritten})

    base_system = system_prompt or DEFAULT_SYSTEM_PROMPT
    if _has_hangul(rewritten or ""):
        base_system += (
            "\n\nCONFIRMED USER LANGUAGE: Korean. "
            "Your final answer MUST be entirely in Korean. No English sentences."
        )
    # rewritten query is self-contained: drop history to keep the agent
    # loop focused and avoid the model mimicking prior assistant prose
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": base_system},
        {"role": "user", "content": rewritten},
    ]
    tool_calls_log: list[dict[str, Any]] = []
    iterations = 0
    language_retries = 0
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
            candidate = choice.get("content") or ""
            # Recovery: qwen2.5-7B occasionally emits a would-be tool call as
            # plain text ('rag_search {"query": "..."}') instead of via the
            # tool_calls channel. Detect that, execute the tool for real, and
            # let the loop continue so the model can synthesize a proper answer.
            recovered = _recover_leaked_tool_call(candidate)
            if recovered is not None:
                name, args_json = recovered
                log.warning("recovered leaked tool_call in content: %s(%s)", name, args_json)
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
                    "id": None,
                    "name": name,
                    "arguments": args_json,
                    "result": result_json,
                    "recovered": True,
                })
                messages.append({"role": "assistant", "content": candidate})
                messages.append({
                    "role": "user",
                    "content": (
                        f"[system note] The previous message was a mis-formatted "
                        f"tool call. I executed `{name}` for you. Result:\n"
                        f"{result_json}\n\nNow write the final answer for the user."
                    ),
                })
                continue
            if (
                _has_hangul(last_user)
                and _has_chinese(candidate)
                and language_retries < 2
            ):
                language_retries += 1
                log.warning("chinese chars in korean answer; forcing rewrite (attempt %d)", language_retries)
                messages.append({"role": "assistant", "content": candidate})
                messages.append({
                    "role": "user",
                    "content": (
                        "[system note] Your previous reply contained Chinese "
                        "characters. Rewrite in KOREAN ONLY. Do not use any "
                        "Han/Chinese characters. If you cannot answer due to "
                        "missing context, reply exactly: "
                        "'죄송합니다. 질문의 맥락이 충분하지 않아 답변드리기 어렵습니다. "
                        "구체적으로 어떤 내용을 확인하고 싶으신지 알려주세요.'"
                    ),
                })
                continue
            if _has_hangul(last_user) and _has_chinese(candidate):
                candidate = "".join(
                    ch for ch in candidate if not ("\u4e00" <= ch <= "\u9fff")
                ).strip()
            final_text = candidate
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


_KNOWN_TOOLS = {t["function"]["name"] for t in tools.TOOLS}


def _recover_leaked_tool_call(text: str) -> tuple[str, str] | None:
    """Detect a would-be tool invocation the model wrote as plain text
    instead of via the tool_calls channel. Returns (name, args_json).

    Handles four observed 7B leak shapes anywhere in the string:
      1. name {"k": v}                              (json object literal)
      2. name(k=v, k=v)                             (python-style kwargs)
      3. samsung_id = name(k=v, ...)                (python assignment prefix)
      4. {"name": "tool", "arguments": {"k": v}}    (openai tool-call wrapper)
    """
    s = (text or "").strip()
    if not s:
        return None
    wrapper = _extract_openai_wrapper(s)
    if wrapper is not None:
        return wrapper
    for name in _KNOWN_TOOLS:
        idx = 0
        while True:
            hit = s.find(name, idx)
            if hit < 0:
                break
            before_ok = hit == 0 or not (s[hit - 1].isalnum() or s[hit - 1] == "_")
            after = s[hit + len(name):hit + len(name) + 1]
            if before_ok and after in ("(", " ", "\n"):
                tail = s[hit + len(name):].lstrip()
                if tail.startswith("{"):
                    parsed = _extract_json_object(tail)
                    if parsed is not None:
                        return name, parsed
                if tail.startswith("("):
                    parsed = _extract_kwargs_as_json(tail)
                    if parsed is not None:
                        return name, parsed
            idx = hit + len(name)
    return None


def _extract_json_object(s: str) -> str | None:
    depth = 0
    end = -1
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    candidate = s[:end]
    try:
        json.loads(candidate)
    except (ValueError, TypeError):
        return None
    return candidate


def _extract_openai_wrapper(s: str) -> tuple[str, str] | None:
    idx = 0
    while True:
        brace = s.find("{", idx)
        if brace < 0:
            return None
        obj = _extract_json_object(s[brace:])
        if obj is not None:
            try:
                data = json.loads(obj)
            except (ValueError, TypeError):
                data = None
            if isinstance(data, dict):
                name = data.get("name")
                args = data.get("arguments")
                if name in _KNOWN_TOOLS and isinstance(args, dict):
                    return name, json.dumps(args, ensure_ascii=False)
            idx = brace + len(obj)
        else:
            idx = brace + 1


def _extract_kwargs_as_json(s: str) -> str | None:
    depth = 0
    end = -1
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    inner = s[1:end - 1].strip()
    if not inner:
        return "{}"
    args: dict[str, Any] = {}
    for pair in _split_top_level_commas(inner):
        if "=" not in pair:
            return None
        key, _, val = pair.partition("=")
        key = key.strip()
        val = val.strip().rstrip(",").strip()
        if not key.isidentifier():
            return None
        try:
            args[key] = json.loads(val)
        except (ValueError, TypeError):
            val_stripped = val.strip('"').strip("'")
            args[key] = val_stripped
    return json.dumps(args, ensure_ascii=False)


def _split_top_level_commas(s: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    in_str: str | None = None
    for ch in s:
        if in_str:
            buf.append(ch)
            if ch == in_str and (not buf or len(buf) < 2 or buf[-2] != "\\"):
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


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
    if name == "rag_search":
        q = args.get("query", "")
        return f"`rag_search` 도구로 RAG 지식베이스에서 **{q}** 를 검색합니다."
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
    if name == "rag_search" and isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
        if not items:
            return "RAG 지식베이스에서 관련 문서를 찾지 못했습니다."
        srcs = list({(it.get("source") or "?") for it in items[:5]})
        return f"RAG에서 {len(items)}건의 관련 passage를 찾았습니다 (출처: {', '.join(srcs[:3])})."
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
