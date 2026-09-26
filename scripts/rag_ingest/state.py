"""State file management (JSON sidecar tracking ingested docs by hash).

Extracted from rag_sync.py so it can be reused across CLI subcommands.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any


def sha256(text: str) -> str:
    """SHA256 hex digest of text (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_state(state_file: pathlib.Path) -> dict[str, Any]:
    """Load state JSON, returning empty structure if missing."""
    if not state_file.exists():
        return {"ingested": {}}
    return json.loads(state_file.read_text(encoding="utf-8"))


def save_state(state_file: pathlib.Path, state: dict[str, Any]) -> None:
    """Atomically write state JSON (temp file + rename)."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(state_file)
