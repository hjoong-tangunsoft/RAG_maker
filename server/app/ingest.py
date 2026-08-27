"""Document loaders: raw text, PDF, plain URLs."""
from __future__ import annotations

import io
import logging
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader

log = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30.0
_MAX_BYTES = 25 * 1024 * 1024  # 25 MB safety cap


def load_pdf_bytes(data: bytes) -> str:
    """Extract text from a PDF byte buffer."""
    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            parts.append(page.extract_text() or "")
        except Exception as e:  # noqa: BLE001
            log.warning("pdf page %d extract failed: %s", i, e)
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def load_text_bytes(data: bytes) -> str:
    """Decode a byte buffer as UTF-8 text (with fallback)."""
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_file(filename: str, data: bytes) -> str:
    """Route by filename extension."""
    if len(data) > _MAX_BYTES:
        raise ValueError(f"file too large: {len(data)} bytes (max {_MAX_BYTES})")
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return load_pdf_bytes(data)
    # treat everything else as text (md, txt, py, json, csv, ...)
    return load_text_bytes(data)


def load_url(url: str) -> tuple[str, str]:
    """Fetch a URL and return (source_label, text_content)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported url scheme: {parsed.scheme}")
    with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        r = client.get(url, headers={"User-Agent": "rag-ingest/1.0"})
        r.raise_for_status()
        content_type = r.headers.get("content-type", "").lower()
        data = r.content
        if len(data) > _MAX_BYTES:
            raise ValueError(f"url content too large: {len(data)} bytes")
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            text = load_pdf_bytes(data)
        else:
            text = load_text_bytes(data)
    label = parsed.netloc + parsed.path
    return label, text
