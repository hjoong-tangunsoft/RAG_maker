"""FastAPI application: RAG endpoints + OpenAI-compatible passthrough."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

from . import embed, ingest, llm, rag
from .ontology.router import router as ontology_router
from .router import router_router
from .services.rag_chat import rag_chat_service
from .chunker import chunk_text
from .config import settings
from .jira_meta import parse_jira_metadata
from .schemas import (
    ChatCompletionRequest,
    DocInfo,
    IngestResponse,
    IngestTextRequest,
    IngestURLRequest,
    QueryRequest,
    QueryResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    TeachRequest,
    TeachResponse,
)
from .store import get_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("rag")

# Silence known-benign Chroma noise:
# - local_persistent_hnsw "Add of existing embedding ID" fires on every query
#   that touches IDs which were double-persisted at ingest time. It's a data
#   artefact from a past double-add of pool-jira-MAN-0198/0199, not a bug in
#   the query path, and does not affect retrieval.
# - product.posthog "capture() takes 1 positional argument but 3 were given"
#   is a Chroma <-> posthog SDK mismatch; telemetry we don't want anyway.
logging.getLogger("chromadb.segment.impl.vector.local_persistent_hnsw").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("warming up embedding model...")
    d = embed.warmup()
    log.info("embedding model ready (dim=%d)", d)
    log.info("opening vector store at %s", settings.chroma_dir)
    stats = get_store().stats()
    log.info("vector store ready: %s", stats)
    yield
    log.info("shutdown")


app = FastAPI(title="RAG service", version="1.0.0", lifespan=lifespan)
app.include_router(ontology_router)
app.include_router(router_router)


# ---------- auth dependency ----------

def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if settings.api_key is None:
        return  # auth disabled
    if x_api_key != settings.api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")


AuthDep = Depends(require_key)


# ---------- health / stats ----------

@app.get("/rag/health")
async def health() -> dict[str, Any]:
    upstream = await llm.health()
    return {
        "status": "ok",
        "upstream_llm": "ok" if upstream else "unreachable",
        "embed_model": settings.embed_model_name,
        "embed_dim": embed.dim(),
    }


@app.get("/rag/stats", dependencies=[AuthDep])
async def stats() -> StatsResponse:
    s = get_store().stats()
    return StatsResponse(
        doc_count=s["doc_count"],
        chunk_count=s["chunk_count"],
        embed_model=settings.embed_model_name,
        llm_model=settings.default_model,
        chroma_dir=str(settings.chroma_dir),
    )


# ---------- docs registry ----------

@app.get("/rag/docs", dependencies=[AuthDep])
async def list_docs() -> list[DocInfo]:
    return [DocInfo(**{k: d[k] for k in ("doc_id", "source", "chunks", "added_at", "bytes")})
            for d in get_store().list_docs()]


@app.delete("/rag/docs/{doc_id}", dependencies=[AuthDep])
async def delete_doc(doc_id: str) -> dict[str, Any]:
    removed = get_store().delete(doc_id)
    if removed == 0 and get_store().get_doc(doc_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "doc not found")
    return {"doc_id": doc_id, "removed_chunks": removed}


# ---------- ingestion ----------

def _index(text: str, source: str, doc_id: str | None, metadata: dict[str, Any]) -> IngestResponse:
    # Phase D-1: auto-extract structured Jira metadata from exporter output
    # so retrieval can filter by status (skip completed) and sort by recency.
    # Non-Jira documents return None and pass through unchanged.
    if jira_meta := parse_jira_metadata(text):
        # Merge Jira fields first, then client-supplied metadata overrides.
        # This preserves explicit caller intent while filling gaps automatically.
        metadata = {**jira_meta, **metadata}

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no content after chunking")
    vecs = embed.embed_passages(chunks)
    doc_id, n = get_store().add(
        doc_id=doc_id,
        source=source,
        chunks=chunks,
        embeddings=vecs,
        base_metadata=metadata,
    )
    return IngestResponse(doc_id=doc_id, source=source, chunks=n, bytes=len(text.encode("utf-8")))


@app.post("/rag/ingest/text", dependencies=[AuthDep])
async def ingest_text(body: IngestTextRequest) -> IngestResponse:
    source = body.source or f"text:{uuid.uuid4().hex[:8]}"
    return _index(body.text, source, body.doc_id, body.metadata)


@app.post("/rag/ingest/file", dependencies=[AuthDep])
async def ingest_file(
    file: UploadFile = File(...),
    doc_id: str | None = Form(None),
    source: str | None = Form(None),
) -> IngestResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty upload")
    try:
        text = ingest.load_file(file.filename or "upload.bin", raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"parse failed: {e}") from e
    if not text.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no extractable text")
    src = source or (file.filename or "upload")
    return _index(text, src, doc_id, {"content_type": file.content_type or "unknown"})


@app.post("/rag/ingest/url", dependencies=[AuthDep])
async def ingest_url(body: IngestURLRequest) -> IngestResponse:
    try:
        label, text = ingest.load_url(body.url)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"fetch failed: {e}") from e
    if not text.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no extractable text")
    return _index(text, label, body.doc_id, {"url": body.url, **body.metadata})


@app.post("/rag/teach", dependencies=[AuthDep])
async def teach_endpoint(body: TeachRequest) -> TeachResponse:
    """Explicit teach endpoint (Path 3, bypasses natural-language trigger).

    Same underlying storage as chat trigger. Tagged with
    strategy='explicit-api' so admin queries can filter it separately from
    both chat-teach and batch ingest.
    """
    from . import teach as _teach
    try:
        result = _teach.auto_ingest(
            content=body.content,
            trigger="explicit-api",
            strategy="explicit-api",
            source_override=body.source,
            metadata_override={
                "explicit_teach": True,
                **(body.metadata or {}),
            },
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    return TeachResponse(
        doc_id=result["doc_id"],
        chunks=result["chunks"],
        bytes=result["bytes"],
        source=result["source"],
        strategy=result["strategy"],
        trigger=result["trigger"],
    )


# ---------- retrieval ----------

@app.post("/rag/search", dependencies=[AuthDep])
async def search(body: SearchRequest) -> SearchResponse:
    hits = rag.retrieve(body.query, k=body.k, where=body.filter)
    return SearchResponse(
        query=body.query,
        hits=[
            SearchHit(
                doc_id=str((h.get("metadata") or {}).get("doc_id", "")),
                chunk_id=str(h.get("chunk_id", "")),
                source=str((h.get("metadata") or {}).get("source", "")),
                score=float(h.get("score", 0.0)),
                text=str(h.get("text", "")),
                metadata=h.get("metadata") or {},
            )
            for h in hits
        ],
    )


@app.post("/rag/query", dependencies=[AuthDep])
async def query(body: QueryRequest) -> QueryResponse:
    text, citations, used_model = await rag.answer(
        query=body.query,
        k=body.k,
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        system_prompt=body.system_prompt,
        where=body.filter,
    )
    return QueryResponse(query=body.query, answer=text, citations=citations, model=used_model)


# ---------- OpenAI-compatible passthrough ----------

@app.get("/rag/v1/models", dependencies=[AuthDep])
async def models() -> Any:
    return await llm.list_models()


@app.post("/rag/v1/chat/completions", dependencies=[AuthDep])
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    """OpenAI-compatible chat completion with RAG injection.

    Body extracted to `services.rag_chat.rag_chat_service` in Issue #37
    Step 4 so the intent router can dispatch to it as a plain Python call
    (no internal HTTP self-call). This wrapper preserves the legacy path
    for backward compat.
    """
    return await rag_chat_service(body, request)


# ---------- debug: raw embedding + pairwise similarity ----------

class SimilarityRequest(BaseModel):
    texts: list[str] = Field(..., min_length=2, max_length=32)


class SimilarityResponse(BaseModel):
    texts: list[str]
    similarity: list[list[float]]
    dim: int


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)
    kind: str = Field(default="passage", pattern="^(passage|query)$")


class EmbedResponse(BaseModel):
    dim: int
    kind: str
    vectors: list[list[float]]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # E5 vectors are L2-normalized


@app.post("/rag/debug/similarity", dependencies=[AuthDep])
async def debug_similarity(body: SimilarityRequest) -> SimilarityResponse:
    vecs = embed.embed_passages(body.texts)
    n = len(vecs)
    matrix = [[round(_cosine(vecs[i], vecs[j]), 6) for j in range(n)] for i in range(n)]
    return SimilarityResponse(texts=body.texts, similarity=matrix, dim=embed.dim())


@app.post("/rag/debug/embed", dependencies=[AuthDep])
async def debug_embed(body: EmbedRequest) -> EmbedResponse:
    vecs = (
        embed.embed_passages(body.texts) if body.kind == "passage"
        else [embed.embed_query(t) for t in body.texts]
    )
    return EmbedResponse(dim=embed.dim(), kind=body.kind, vectors=vecs)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    dur_ms = int((time.perf_counter() - start) * 1000)
    log.info("%s %s -> %d (%d ms)", request.method, request.url.path, response.status_code, dur_ms)
    return response
