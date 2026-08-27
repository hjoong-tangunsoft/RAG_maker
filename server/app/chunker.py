"""Text chunker: paragraph-aware, size-bounded, Korean-safe."""
from __future__ import annotations

import re
from .config import settings


# Prefer these separators, in order, when splitting oversized blocks.
# Includes CJK-friendly sentence terminators.
_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "。 ",
    ". ",
    "! ",
    "? ",
    "。",
    ". ",
    "; ",
    ", ",
    " ",
    "",
)


def normalize(text: str) -> str:
    """Strip weird whitespace but preserve paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse >2 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_by(text: str, sep: str) -> list[str]:
    if sep == "":
        return list(text)
    parts = text.split(sep)
    # re-attach the separator (except the last)
    result = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            result.append(p + sep)
        elif p:
            result.append(p)
    return [x for x in result if x]


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily merge pieces into chunks of <= size chars with overlap."""
    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if len(buf) + len(piece) <= size:
            buf += piece
            continue
        if buf:
            chunks.append(buf)
        # start new buffer, keep tail of previous as overlap
        if overlap and chunks:
            tail = chunks[-1][-overlap:]
            buf = tail + piece
        else:
            buf = piece
        # if a single piece is already oversized, hard-split it later
        if len(buf) > size * 2:
            chunks.append(buf[:size])
            buf = buf[size - overlap :] if overlap else buf[size:]
    if buf:
        chunks.append(buf)
    return chunks


def chunk_text(
    text: str,
    size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks.

    Uses a recursive-separator strategy similar to LangChain's
    RecursiveCharacterTextSplitter, tuned for CJK + English mix.
    """
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    text = normalize(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    # Try separators in priority order until pieces are small enough
    pieces = [text]
    for sep in _SEPARATORS:
        pieces = [p for chunk in pieces for p in _split_by(chunk, sep)]
        if all(len(p) <= size for p in pieces):
            break

    chunks = _merge(pieces, size=size, overlap=overlap)
    # final safety: hard-cut anything still oversized
    out: list[str] = []
    for c in chunks:
        if len(c) <= size:
            out.append(c)
        else:
            for i in range(0, len(c), size - overlap):
                out.append(c[i : i + size])
    return [c.strip() for c in out if c.strip()]
