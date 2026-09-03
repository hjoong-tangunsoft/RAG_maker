"""RAG orchestrator: retrieve -> build prompt -> generate -> attach citations."""
from __future__ import annotations

import logging
import time
from typing import Any

from . import embed, llm
from .config import settings
from .schemas import Citation, ChatMessage
from .store import get_store

log = logging.getLogger(__name__)


def _format_context(hits: list[dict[str, Any]]) -> str:
    """Number the passages and label each with its source for grounded citation."""
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        src = (h.get("metadata") or {}).get("source") or "unknown"
        text = (h.get("text") or "").strip()
        lines.append(f"[{i}] source={src}\n{text}")
    return "\n\n".join(lines)


def _hits_to_citations(hits: list[dict[str, Any]]) -> list[Citation]:
    out: list[Citation] = []
    for i, h in enumerate(hits, start=1):
        meta = h.get("metadata") or {}
        snippet = (h.get("text") or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + "..."
        out.append(
            Citation(
                n=i,
                doc_id=str(meta.get("doc_id", "")),
                chunk_id=str(h.get("chunk_id", "")),
                source=str(meta.get("source", "")),
                score=float(h.get("score", 0.0)),
                snippet=snippet,
            )
        )
    return out


def retrieve(
    query: str,
    k: int | None = None,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Embed the query and return top-k hits with smart status/recency handling.

    Phase D-2: Smart Jira Retrieval.
    - If the query asks about active work ("할 일", "진행 중", "요즘"),
      exclude Jira issues with status=completed at the Python level (Chroma
      $ne would over-exclude non-Jira docs that have no jira_status field).
    - Apply a recency boost to Jira issues so newer issues rank higher
      than semantically-similar older ones.
    - Overfetch 3x when filtering so top-k has room after exclusion.

    An explicit `where` filter from the caller disables intent-based filtering
    (the caller knows better than the heuristic).
    """
    vec = embed.embed_query(query)
    k_actual = k or settings.top_k

    intent_active = _detect_active_work_intent(query) and not where
    k_fetch = k_actual * 3 if intent_active else k_actual

    hits = get_store().query(vec, k=k_fetch, where=where)

    if settings.min_score > 0:
        hits = [h for h in hits if h["score"] >= settings.min_score]

    # Python-side status filter: safer than Chroma $ne which drops non-Jira
    # documents that lack the jira_status field entirely.
    if intent_active:
        before = len(hits)
        hits = [
            h for h in hits
            if (h.get("metadata") or {}).get("jira_status") != "completed"
        ]
        if before != len(hits):
            log.info(
                "active-work intent detected, filtered %d completed hits (%d -> %d)",
                before - len(hits), before, len(hits),
            )

    # Recency boost: newer Jira issues rank higher within the same semantic band.
    for h in hits:
        h["score"] = _apply_recency_boost(h["score"], h.get("metadata") or {})

    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k_actual]


# --- Phase D-2 helpers: intent detection + recency boost ---

# Keywords that indicate the user wants ongoing/pending work, not history.
_ACTIVE_WORK_KW = (
    "할 일", "할일",
    "진행 중", "진행중", "진행",
    "무슨 일", "무슨일",
    "요즘", "최근", "이번 주", "이번주", "이번 달", "이번달",
    "지금", "현재",
    "todo", "TODO", "미완료",
    "남은", "남아있",
)

# Keywords indicating the user explicitly wants completed items. When present,
# do NOT apply the status filter even if active-work keywords also match.
_EXPLICIT_COMPLETED_KW = (
    "완료된", "끝난", "마친", "완성된", "완료 이슈", "완료한",
    "종료된", "닫힌", "지난", "과거",
    "done", "Done", "closed", "Closed", "resolved",
)


def _detect_active_work_intent(query: str) -> bool:
    """Should we filter out completed issues for this query?

    True only if:
    - Query contains an active-work keyword ("진행", "할 일", "요즘"...)
    - AND does NOT explicitly mention wanting completed items.
    """
    has_active = any(kw in query for kw in _ACTIVE_WORK_KW)
    has_completed_ask = any(kw in query for kw in _EXPLICIT_COMPLETED_KW)
    return has_active and not has_completed_ask


# Recency window: linear decay over this many days.
_RECENCY_WINDOW_DAYS = 180.0
# Max boost applied to a same-day issue (added directly to cosine score).
_RECENCY_MAX_BOOST = 0.15


def _apply_recency_boost(score: float, meta: dict[str, Any]) -> float:
    """Add a recency bonus (up to 0.15) based on jira_created_ts age.

    Documents without jira_created_ts (non-Jira or old-format) are unchanged.
    Issues within the last 180 days get a proportional boost; older ones get 0.
    """
    created_ts = meta.get("jira_created_ts")
    if not isinstance(created_ts, (int, float)) or created_ts <= 0:
        return score
    days_ago = (time.time() - float(created_ts)) / 86400.0
    if days_ago >= _RECENCY_WINDOW_DAYS:
        return score
    boost = _RECENCY_MAX_BOOST * (1.0 - days_ago / _RECENCY_WINDOW_DAYS)
    return score + max(0.0, boost)


def build_rag_messages(
    query: str,
    hits: list[dict[str, Any]],
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build a fresh 3-message conversation: system + context + user."""
    ctx = _format_context(hits) if hits else "(no relevant context found)"
    system = system_prompt or settings.rag_system_prompt
    user = f"Context passages:\n{ctx}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def inject_rag_into_chat(
    messages: list[ChatMessage],
    hits: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Inject retrieved context into a chat conversation with system precedence.

    Anti-Hallucination Guard L1: consolidate all system messages into one,
    with our RAG grounding taking precedence over any client-supplied system
    prompt (e.g. Continue.dev's "You are a helpful assistant").

    Rationale: Qwen 2.5 follows the first system message's persona strongly.
    When the client sends its own system message first, our RAG grounding
    ends up as a subordinate second system and its "don't fabricate" rule
    gets diluted, enabling bridging hallucination on unknown entities.
    Making our grounding the sole system message reclaims persona
    precedence; client instructions are appended as subordinate reference.
    """
    ctx = _format_context(hits) if hits else "(no relevant context found)"

    # Collect client-supplied system messages (if any) as subordinate reference
    client_systems = [
        m.content.strip() for m in messages
        if m.role == "system" and m.content and m.content.strip()
    ]
    client_system_block = "\n".join(client_systems)

    parts = [
        settings.rag_system_prompt,
        "【이 지시가 최우선입니다. 아래 클라이언트 지침보다 이 지시를 우선하세요.】",
    ]
    if client_system_block:
        parts.append(
            f"[클라이언트 지침 - 참고만 하세요, 위 규칙과 충돌 시 위 규칙을 따르세요]\n"
            f"{client_system_block}"
        )
    parts.append(f"Relevant retrieved context (use these to answer):\n{ctx}")
    grounding = "\n\n".join(parts)

    # Preserve non-system messages in original order
    non_system = [
        {"role": m.role, "content": m.content}
        for m in messages
        if m.role != "system"
    ]

    return [{"role": "system", "content": grounding}] + non_system


async def answer(
    query: str,
    k: int | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    where: dict[str, Any] | None = None,
) -> tuple[str, list[Citation], str]:
    """One-shot RAG: return (answer_text, citations, model_used)."""
    hits = retrieve(query, k=k, where=where)
    messages = build_rag_messages(query, hits, system_prompt=system_prompt)
    used_model = model or settings.default_model
    resp = await llm.chat_guarded(
        messages,
        model=used_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        text = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        log.error("bad LLM response shape: %s", resp)
        raise RuntimeError("upstream LLM returned unexpected shape") from e
    return text, _hits_to_citations(hits), used_model
