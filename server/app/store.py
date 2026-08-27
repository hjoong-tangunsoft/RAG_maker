"""Chroma-backed vector store: docs collection + doc metadata registry."""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings

log = logging.getLogger(__name__)

_COLL_NAME = "rag_chunks"


class VectorStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        # cosine distance -> similarity = 1 - distance
        self.collection = self.client.get_or_create_collection(
            name=_COLL_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._registry: Path = settings.data_dir / "docs.json"
        if not self._registry.exists():
            self._registry.write_text("{}", encoding="utf-8")

    # ----- doc registry (sidecar json, keyed by doc_id) -----

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self._registry.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_registry(self, reg: dict[str, dict[str, Any]]) -> None:
        tmp = self._registry.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._registry)

    def list_docs(self) -> list[dict[str, Any]]:
        reg = self._load_registry()
        return sorted(reg.values(), key=lambda d: d.get("added_at", ""), reverse=True)

    def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        return self._load_registry().get(doc_id)

    # ----- write -----

    def add(
        self,
        doc_id: str | None,
        source: str,
        chunks: list[str],
        embeddings: list[list[float]],
        base_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        doc_id = doc_id or uuid.uuid4().hex
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return doc_id, 0

        # replace any prior version of this doc
        self.delete(doc_id, prune_registry=False)

        base_metadata = base_metadata or {}
        ts = int(time.time())
        ids = [f"{doc_id}::{i:05d}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "chunk_index": i,
                "source": source,
                "added_at": ts,
                **base_metadata,
            }
            for i in range(len(chunks))
        ]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        reg = self._load_registry()
        reg[doc_id] = {
            "doc_id": doc_id,
            "source": source,
            "chunks": len(chunks),
            "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
            "bytes": sum(len(c.encode("utf-8")) for c in chunks),
            "metadata": base_metadata,
        }
        self._save_registry(reg)
        log.info("added doc %s (%d chunks) from %s", doc_id, len(chunks), source)
        return doc_id, len(chunks)

    def delete(self, doc_id: str, prune_registry: bool = True) -> int:
        try:
            existing = self.collection.get(where={"doc_id": doc_id})
            ids = existing.get("ids", []) or []
        except Exception:  # noqa: BLE001
            ids = []
        if ids:
            self.collection.delete(ids=ids)
        if prune_registry:
            reg = self._load_registry()
            reg.pop(doc_id, None)
            self._save_registry(reg)
        return len(ids)

    # ----- read -----

    def query(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits: list[dict[str, Any]] = []
        for i, chunk_id in enumerate(ids):
            distance = float(dists[i]) if i < len(dists) else 1.0
            similarity = max(0.0, 1.0 - distance)  # cosine distance -> similarity
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": similarity,
                }
            )
        return hits

    def stats(self) -> dict[str, int]:
        return {
            "chunk_count": self.collection.count(),
            "doc_count": len(self._load_registry()),
        }


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
