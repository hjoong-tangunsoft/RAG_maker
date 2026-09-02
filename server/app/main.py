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
from fastapi.responses import JSONResponse, StreamingResponse

from pydantic import BaseModel, Field

from . import embed, ingest, llm, rag
from .chunker import chunk_text
from .config import settings
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
)
from .store import get_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("rag")


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
    # If rag=false, straight passthrough
    messages_out: list[dict[str, str]]
    hits: list[dict[str, Any]] = []
    if body.rag:
        # find the last user message to use as the retrieval query
        last_user = next((m.content for m in reversed(body.messages) if m.role == "user"), None)
        if last_user:
            hits = rag.retrieve(last_user, k=body.rag_k, where=body.rag_filter)
        messages_out = rag.inject_rag_into_chat(body.messages, hits)
    else:
        messages_out = [{"role": m.role, "content": m.content} for m in body.messages]

    if body.stream:
        async def gen():
            async for chunk in llm.stream_chat(
                messages_out,
                model=body.model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            ):
                yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")

    resp = await llm.chat_guarded(
        messages_out,
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )
    # attach citations to response for clients that care
    if body.rag and hits:
        resp["citations"] = [c.model_dump() for c in rag._hits_to_citations(hits)]
    return JSONResponse(resp)


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
