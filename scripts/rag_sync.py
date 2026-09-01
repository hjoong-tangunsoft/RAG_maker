#!/usr/bin/env python3
"""RAG daily sync - Dumb Server, Smart Client pipeline.

Reads .md files from POOL_DIR, ingests up to DAILY_LIMIT new/changed
documents per run via /rag/ingest/text. State (hash) tracked in STATE_FILE.

Follows the "Dumb Server, Smart Client" principle documented in
docs/DESIGN_PRINCIPLES.md — the server just stores; this client does
hash-based dedup, action classification (NEW/CHANGED/UNCHANGED), and
throttled ingestion.

Meant to be run via systemd timer (rag-sync.timer, daily 03:00 KST).

Environment variables:
    POOL_DIR      Source docs directory   (default: /upload/rag/data/docs-pool)
    STATE_FILE    JSON state file         (default: /upload/rag/data/sync-state.json)
    RAG_URL       Ingest endpoint         (default: http://127.0.0.1:8100/rag/ingest/text)
    DAILY_LIMIT   Max ingests per run     (default: 2)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import sys
from datetime import datetime, timezone

import httpx

POOL_DIR = pathlib.Path(os.environ.get("POOL_DIR", "/upload/rag/data/docs-pool"))
STATE_FILE = pathlib.Path(os.environ.get("STATE_FILE", "/upload/rag/data/sync-state.json"))
RAG_URL = os.environ.get("RAG_URL", "http://127.0.0.1:8100/rag/ingest/text")
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "2"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("rag-sync")


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"ingested": {}}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_one(md_file: pathlib.Path, text: str) -> dict:
    """POST /rag/ingest/text. Server is dumb - it just stores what we send."""
    doc_id = f"pool-{md_file.stem}"
    payload = {
        "doc_id": doc_id,
        "source": md_file.name,
        "text": text,
        "metadata": {
            "pool_path": str(md_file),
            "hash": sha256(text),
            "ingested_by": "rag-sync",
        },
    }
    with httpx.Client(timeout=60.0) as c:
        r = c.post(RAG_URL, json=payload)
        r.raise_for_status()
    return r.json()


def main() -> int:
    if not POOL_DIR.exists():
        log.error("pool dir missing: %s", POOL_DIR)
        return 1

    state = load_state()
    ingested = state.setdefault("ingested", {})

    all_docs = sorted(POOL_DIR.glob("*.md"))
    log.info("pool has %d docs, %d already tracked", len(all_docs), len(ingested))

    # Classify each file: NEW / CHANGED / UNCHANGED
    todo = []
    for md_file in all_docs:
        text = md_file.read_text(encoding="utf-8")
        h = sha256(text)
        prev = ingested.get(md_file.name)
        if prev is None:
            todo.append((md_file, text, h, "NEW"))
        elif prev["hash"] != h:
            todo.append((md_file, text, h, "CHANGED"))
        # else: UNCHANGED - skip silently

    if not todo:
        log.info("nothing to sync (all up to date)")
        return 0

    log.info("%d pending, will process up to %d this run", len(todo), DAILY_LIMIT)

    done = 0
    failed = 0
    for md_file, text, h, action in todo[:DAILY_LIMIT]:
        try:
            resp = ingest_one(md_file, text)
            ingested[md_file.name] = {
                "hash": h,
                "doc_id": resp["doc_id"],
                "chunks": resp["chunks"],
                "action": action,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            log.info(
                "  %-8s %s -> %d chunks (%d bytes)",
                action, md_file.name, resp["chunks"], resp["bytes"],
            )
            done += 1
        except Exception as e:  # noqa: BLE001
            log.error("  FAIL     %s: %s", md_file.name, e)
            failed += 1

    save_state(state)
    log.info(
        "sync complete: %d done, %d failed, %d still pending, %d total tracked",
        done, failed, len(todo) - done, len(ingested),
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
