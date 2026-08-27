"""Embedding model wrapper (multilingual E5, CPU)."""
from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Lazy-load the embedding model; cache the singleton."""
    log.info("Loading embedding model: %s", settings.embed_model_name)
    model = SentenceTransformer(
        settings.embed_model_name,
        cache_folder=str(settings.models_dir),
        device="cpu",
    )
    # E5 recommends L2-normalized output for cosine similarity via dot product
    log.info("Embedding model loaded. dim=%d", model.get_sentence_embedding_dimension())
    return model


def warmup() -> int:
    """Force model load at startup. Returns embedding dimension."""
    m = _model()
    # small dummy pass to compile any kernels
    _ = m.encode(["warmup"], normalize_embeddings=True, show_progress_bar=False)
    return int(m.get_sentence_embedding_dimension())


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Encode documents/passages with the passage prefix."""
    if not texts:
        return []
    prefixed = [f"{settings.embed_passage_prefix}{t}" for t in texts]
    vecs = _model().encode(
        prefixed,
        batch_size=settings.embed_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    """Encode a single query with the query prefix."""
    prefixed = f"{settings.embed_query_prefix}{text}"
    vec = _model().encode(
        [prefixed],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    return vec.tolist()


def dim() -> int:
    return int(_model().get_sentence_embedding_dimension())
