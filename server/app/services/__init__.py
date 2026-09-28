"""Extracted chat-completion service functions (Issue #37 Step 4).

The full pipelines for /rag/v1/chat/completions and /ontology/v1/chat/completions
live here so the intent router can dispatch to them as plain Python
awaits (no internal HTTP self-call). The original endpoints in main.py /
ontology/router.py remain as thin wrappers for backward compat.
"""
