"""Intent router package (Issue #37, P5 v3).

Provides POST /router/v1/chat/completions which classifies user intent
(plain/rag/ontology) via LLM tool-calling and dispatches to the
appropriate service. Legacy /rag/v1 and /ontology/v1 endpoints remain
for backward compat (deprecated but supported).

See docs/notion-drafts/2026-09-27-litellm-vs-intent-router.md for
architecture rationale.
"""
from .router import router as router_router

__all__ = ["router_router"]
