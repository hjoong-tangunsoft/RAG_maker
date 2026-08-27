"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="/upload/rag/rag.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Filesystem layout
    data_dir: Path = Path("/upload/rag/data")
    chroma_dir: Path = Path("/upload/rag/data/chroma")
    docs_dir: Path = Path("/upload/rag/data/docs")
    models_dir: Path = Path("/upload/rag/models")

    # Server bind
    host: str = "127.0.0.1"
    port: int = 8100

    # Auth: if set, every request except /health must send X-API-Key
    api_key: str | None = None

    # Downstream LLM (LiteLLM proxy)
    litellm_url: str = "http://127.0.0.1:4000"
    litellm_api_key: str = "sk-1234"  # matches LITELLM_MASTER_KEY
    default_model: str = "qwen2.5-7b"

    # Embedding model (CPU)
    embed_model_name: str = "intfloat/multilingual-e5-base"
    embed_batch_size: int = 32
    # E5 family requires prefixes for asymmetric passage/query encoding
    embed_query_prefix: str = "query: "
    embed_passage_prefix: str = "passage: "

    # Chunking
    chunk_size: int = 800  # chars
    chunk_overlap: int = 120  # chars

    # Retrieval
    top_k: int = 5
    min_score: float = 0.0  # cosine similarity threshold (0..1); 0 = no filter

    # Generation
    rag_temperature: float = 0.2
    rag_max_tokens: int = 1024
    rag_system_prompt: str = (
        "You are a precise assistant. Answer the user's question using ONLY the "
        "provided context. If the context is insufficient or irrelevant, say so "
        "honestly instead of guessing. Always cite sources as [n] where n matches "
        "the numbered context passage you used. Respond in the same language as "
        "the user's question."
    )


settings = Settings()

# Ensure runtime directories exist
for p in (settings.data_dir, settings.chroma_dir, settings.docs_dir, settings.models_dir):
    p.mkdir(parents=True, exist_ok=True)
