"""자연어 학습 트리거 감지 + 자동 인제스트 (Path 3 Ingest).

사용자가 채팅 중 '학습해/저장해/기억해' 같은 명령형 어미를 사용하면
자동으로 Chroma 에 저장. Server stays dumb - detection and content
extraction happen client-side (in chat middleware); the storage endpoint
just receives ingest calls like any other path.

Follows the 'Dumb Server, Smart Client' principle from
docs/DESIGN_PRINCIPLES.md.

Endpoints:
- POST /rag/v1/chat/completions triggers this middleware
- POST /rag/teach is an explicit endpoint (main.py) that also uses auto_ingest()
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any

from .chunker import chunk_text
from .embed import embed_passages
from .schemas import ChatMessage
from .store import get_store

log = logging.getLogger(__name__)

# ---- Trigger patterns ----
#
# Match imperative endings only ("학습해", "저장해줘", "기억해라").
# Avoid false positives on declarative/negative forms
# ("학습됐다", "저장된 문서", "기억이 안 나").

_TEACH_PATTERNS = [
    # Korean natural language - imperative endings required
    re.compile(r"(학습|저장|기억|인제스트|공부|노트)\s*(해|해줘|해라|해둬|해주세요|부탁|줘)"),
    # RAG-specific phrasings
    re.compile(r"RAG\s*에(?:다|다가)?\s*(넣어|저장|추가)"),
    # Slash / at-commands
    re.compile(r"^\s*/(save|learn|remember|teach)\b", re.IGNORECASE),
    re.compile(r"@(save|learn|remember|teach)\b", re.IGNORECASE),
    # English natural language
    re.compile(r"\bremember\s+this\b", re.IGNORECASE),
    re.compile(r"\bsave\s+(this|it)\b", re.IGNORECASE),
    re.compile(r"\blearn\s+this\b", re.IGNORECASE),
]

# Tier 1 - explicit marker
_MARKER_RE = re.compile(
    r"---\s*(?:LEARN|학습(?:시작)?|저장시작)\s*---(.*?)---\s*(?:END|학습(?:끝)?|저장끝)\s*---",
    re.DOTALL | re.IGNORECASE,
)

# Minimum content length (avoid spam / accidental short saves)
MIN_CONTENT_LEN = 30


def detect_teach_trigger(text: str) -> str | None:
    """Return the matched trigger phrase, or None if no trigger."""
    if not text:
        return None
    for pattern in _TEACH_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


def find_prev_assistant(messages: list[ChatMessage]) -> str | None:
    """Find the assistant message that came before the last user message.

    Used for Tier 3: user says "방금 답변 저장해" -> save the assistant's
    previous response.
    """
    found_user = False
    for m in reversed(messages):
        if m.role == "user":
            found_user = True
            continue
        if found_user and m.role == "assistant" and m.content:
            return m.content
    return None


def extract_teachable_content(
    current_msg: str,
    trigger: str,
    prev_assistant_msg: str | None = None,
) -> tuple[str, str]:
    """Extract what to save from the user's message.

    Priority (3-tier strategy):
    1. 'marker-block'   - explicit ---학습시작--- ... ---학습끝--- block
    2. 'user-message'   - message with trigger phrase removed (>= MIN_CONTENT_LEN)
    3. 'prev-assistant' - previous assistant response (short-trigger fallback)

    Returns:
        (content_text, strategy_label)

    Raises:
        ValueError if nothing extractable.
    """
    # Tier 1: explicit marker
    m = _MARKER_RE.search(current_msg)
    if m:
        content = m.group(1).strip()
        if len(content) >= MIN_CONTENT_LEN:
            return content, "marker-block"

    # Tier 2: strip trigger + common connectors, use what remains
    without_trigger = current_msg
    for pattern in _TEACH_PATTERNS:
        without_trigger = pattern.sub("", without_trigger)
    # Remove common Korean lead-in phrases
    without_trigger = re.sub(
        r"^\s*(?:이거|이걸|이것을?|이\s*내용을?|위\s*내용을?|"
        r"다음(?:을|를)?|여기\s*내용을?)\s*[:：]?\s*",
        "",
        without_trigger,
    )
    # Trim trailing punctuation
    without_trigger = re.sub(r"[.,!?\s]+$", "", without_trigger).strip()

    if len(without_trigger) >= MIN_CONTENT_LEN:
        return without_trigger, "user-message"

    # Tier 3: previous assistant message
    if prev_assistant_msg and len(prev_assistant_msg.strip()) >= MIN_CONTENT_LEN:
        return prev_assistant_msg.strip(), "prev-assistant"

    raise ValueError("저장할 내용을 찾을 수 없습니다")


def _generate_doc_id(content: str) -> str:
    """Timestamp + short hash for uniqueness + traceability."""
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    return f"user-taught-{ts}-{h}"


def auto_ingest(
    content: str,
    trigger: str,
    strategy: str,
    source_override: str | None = None,
    metadata_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Chunk + embed + store to Chroma. Returns doc info dict.

    Same underlying flow as _index() in main.py, but with teach-specific
    metadata tagging so we can filter later:
        rag-ingest ls --source chat-teach
    """
    if len(content.strip()) < MIN_CONTENT_LEN:
        raise ValueError(f"저장할 내용이 너무 짧습니다 ({len(content)}자)")

    doc_id = _generate_doc_id(content)
    ts_str = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    source = source_override or f"chat-teach:{ts_str}"

    chunks = chunk_text(content)
    if not chunks:
        raise ValueError("청킹 결과 비어있음")

    vecs = embed_passages(chunks)

    base_metadata: dict[str, Any] = {
        "ingested_by": "chat-trigger",
        "trigger_phrase": trigger,
        "extraction_strategy": strategy,
        "auto_ingested": True,
    }
    if metadata_override:
        base_metadata.update(metadata_override)

    final_doc_id, n_chunks = get_store().add(
        doc_id=doc_id,
        source=source,
        chunks=chunks,
        embeddings=vecs,
        base_metadata=base_metadata,
    )

    log.info(
        "teach ingested: doc_id=%s chunks=%d strategy=%s trigger=%r bytes=%d",
        final_doc_id, n_chunks, strategy, trigger, len(content.encode("utf-8")),
    )

    return {
        "doc_id": final_doc_id,
        "chunks": n_chunks,
        "bytes": len(content.encode("utf-8")),
        "source": source,
        "strategy": strategy,
        "trigger": trigger,
    }


def format_confirmation_message(result: dict[str, Any], preview: str) -> str:
    """Format the assistant-facing confirmation message."""
    preview_short = preview.strip()[:120]
    if len(preview.strip()) > 120:
        preview_short += "..."
    return (
        f"✅ **학습 완료** (`{result['trigger']}` 감지)\n\n"
        f"- **doc_id**: `{result['doc_id']}`\n"
        f"- **청크**: {result['chunks']}개\n"
        f"- **크기**: {result['bytes']} bytes\n"
        f"- **저장 방식**: `{result['strategy']}`\n\n"
        f"**미리보기:**\n> {preview_short}\n\n"
        f"앞으로 관련 질문에 이 내용을 참조해서 답변합니다."
    )
