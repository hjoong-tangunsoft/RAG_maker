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
    """Inject retrieved context into an existing chat history.

    Strategy: append a system message *right before* the last user turn so it
    grounds the answer without discarding prior conversation history.
    """
    ctx = _format_context(hits) if hits else "(no relevant context found)"
    grounding = (
        f"{settings.rag_system_prompt}\n\n"
        f"Relevant retrieved context (use these to answer):\n{ctx}"
    )
    grounding_msg = {"role": "system", "content": grounding}

    out: list[dict[str, str]] = []
    injected = False
    # walk from the end to find the last user message
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break

    for i, m in enumerate(messages):
        if i == last_user_idx and not injected:
            out.append(grounding_msg)
            injected = True
        out.append({"role": m.role, "content": m.content})

    if not injected:
        # no user message at all - just prepend
        out.insert(0, grounding_msg)
    return out


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
