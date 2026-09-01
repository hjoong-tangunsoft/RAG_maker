"""RAG document ingest CLI package.

Higher-level wrapper around the same core logic as `scripts/rag_sync.py`
(kept side-by-side for comparison). Adds subcommands, help text, and
common operational shortcuts.

Follows the "Dumb Server, Smart Client" principle documented in
docs/DESIGN_PRINCIPLES.md.
"""
__version__ = "0.1.0"
