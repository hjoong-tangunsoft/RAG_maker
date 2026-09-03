"""RAG orchestrator: retrieve -> build prompt -> generate -> attach citations."""
from __future__ import annotations

import logging
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
    """Embed the query and return top-k hits above min_score."""
    vec = embed.embed_query(query)
    hits = get_store().query(vec, k=k or settings.top_k, where=where)
    if settings.min_score > 0:
        hits = [h for h in hits if h["score"] >= settings.min_score]
    return hits


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
