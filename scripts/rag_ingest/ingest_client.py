"""HTTP client for the RAG server. All server interactions go through here.

Extracted from rag_sync.py to enable multiple subcommands to hit the
same server (ingest, list, delete, health, stats).
"""
from __future__ import annotations

import pathlib
from typing import Any

import httpx

from .state import sha256


class IngestClient:
    """Thin wrapper over the RAG server HTTP API.

    Accepts either a base URL (`http://127.0.0.1:8100`) or the full
    ingest endpoint (`http://127.0.0.1:8100/rag/ingest/text`) - normalizes
    both cases so callers don't need to worry.
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        base = base_url.rstrip("/")
        if base.endswith("/rag/ingest/text"):
            self.rag_root = base.rsplit("/ingest/text", 1)[0]
        elif base.endswith("/rag"):
            self.rag_root = base
        else:
            self.rag_root = f"{base}/rag"
        self.timeout = timeout

    @property
    def ingest_url(self) -> str:
        return f"{self.rag_root}/ingest/text"

    def ingest(self, md_file: pathlib.Path, text: str) -> dict[str, Any]:
        """POST /rag/ingest/text. Server is dumb - it just stores what we send."""
        doc_id = f"pool-{md_file.stem}"
        payload = {
            "doc_id": doc_id,
            "source": md_file.name,
            "text": text,
            "metadata": {
                "pool_path": str(md_file),
                "hash": sha256(text),
                "ingested_by": "rag-ingest",
            },
        }
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(self.ingest_url, json=payload)
            r.raise_for_status()
        return r.json()

    def list_docs(self) -> list[dict[str, Any]]:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(f"{self.rag_root}/docs")
            r.raise_for_status()
        return r.json()

    def stats(self) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(f"{self.rag_root}/stats")
            r.raise_for_status()
        return r.json()

    def delete(self, doc_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.delete(f"{self.rag_root}/docs/{doc_id}")
            r.raise_for_status()
        return r.json()

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as c:
            r = c.get(f"{self.rag_root}/health")
            r.raise_for_status()
        return r.json()
