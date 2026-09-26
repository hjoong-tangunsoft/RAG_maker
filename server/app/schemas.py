"""Pydantic request/response models."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


# ---------- Ingestion ----------

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    doc_id: str | None = None
    source: str | None = None  # human-readable source label
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestURLRequest(BaseModel):
    url: str
    doc_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    source: str
    chunks: int
    bytes: int


# ---------- Retrieval ----------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    filter: dict[str, Any] | None = None


class SearchHit(BaseModel):
    doc_id: str
    chunk_id: str
    source: str
    score: float  # cosine similarity in [0, 1] (approx)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


# ---------- One-shot RAG query ----------

class Citation(BaseModel):
    n: int  # 1-based citation index used in prompt
    doc_id: str
    chunk_id: str
    source: str
    score: float
    snippet: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None
    filter: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    model: str


# ---------- OpenAI-compatible chat ----------

class ToolFunction(BaseModel):
    """Function metadata inside an assistant's tool_call.

    Per OpenAI spec, `arguments` is a JSON-encoded string (not a dict) so
    that streaming can build it up incrementally.
    """
    name: str
    arguments: str


class ToolCall(BaseModel):
    """One tool invocation the LLM decided to make.

    Assistant messages carrying tool_calls have content=None and the
    client is expected to execute each tool and reply with a matching
    tool-role message referencing this id.
    """
    id: str
    type: Literal["function"] = "function"
    function: ToolFunction


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message with tool-calling support.

    Content is optional because assistant messages carrying tool_calls
    have content=None per the OpenAI spec. Tool-role messages carry the
    execution result of a prior tool_call and reference it by
    tool_call_id.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # only on role="tool" messages
    name: str | None = None  # tool name on role="tool" messages


class ChatCompletionRequest(BaseModel):
    """Superset of OpenAI's schema with a `rag` extension and tool_calls passthrough.

    When `tools` is present the server passes them through to the upstream
    LLM unchanged and skips RAG injection (Continue.dev-style agent mode).
    """
    model: str = "qwen2.5-7b"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    # OpenAI tool-calling passthrough (Phase 1 of Issue #23)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    # RAG extension: if true (default when hitting /rag/v1/*), retrieval is applied
    rag: bool = True
    rag_k: int | None = None
    rag_filter: dict[str, Any] | None = None


# ---------- Docs / stats ----------

class DocInfo(BaseModel):
    doc_id: str
    source: str
    chunks: int
    added_at: str
    bytes: int | None = None


class StatsResponse(BaseModel):
    doc_count: int
    chunk_count: int
    embed_model: str
    llm_model: str
    chroma_dir: str


# ---------- Teach (Path 3 - explicit ingest via chat semantics) ----------

class TeachRequest(BaseModel):
    """Explicit teach endpoint request (POST /rag/teach).

    Used by:
    - Sync CLI scripts, admin UI buttons (bypass natural-language trigger)
    - When the exact content boundary needs to be programmatic
    """
    content: str = Field(..., min_length=30)
    source: str | None = None
    doc_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TeachResponse(BaseModel):
    doc_id: str
    chunks: int
    bytes: int
    source: str
    strategy: str  # 'marker-block' | 'user-message' | 'prev-assistant' | 'explicit-api'
    trigger: str  # trigger phrase, or 'explicit-api' if via /rag/teach
