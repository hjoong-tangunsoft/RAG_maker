"""Core sync logic. Same behavior as rag_sync.py but as a reusable function.

Every CLI subcommand that ingests calls `sync()` here. The only difference
from rag_sync.py is packaging: same NEW/CHANGED/UNCHANGED classification,
same hash-based dedup, same idempotency guarantees. CLI adds per-item
timing and title extraction for richer output.
"""
from __future__ import annotations

import logging
import pathlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .ingest_client import IngestClient
from .state import load_state, save_state, sha256

log = logging.getLogger("rag-ingest.core")

_HEADING_RE = re.compile(r"^#\s+(.+?)$", re.MULTILINE)


def extract_title(text: str) -> str:
    """Return the first Markdown heading text, or empty string if none."""
    m = _HEADING_RE.search(text)
    return m.group(1).strip() if m else ""


@dataclass
class SyncItem:
    md_file: pathlib.Path
    text: str
    hash: str
    action: str  # "NEW" | "CHANGED" | "UNCHANGED"
    title: str = ""


@dataclass
class SyncedDoc:
    filename: str
    title: str
    action: str
    chunks: int
    bytes: int
    duration_ms: int


@dataclass
class SyncResult:
    pool_total: int
    tracked_before: int
    done: int
    failed: int
    pending: int
    tracked_after: int
    total_duration_ms: int = 0
    processed: list[SyncedDoc] = field(default_factory=list)


def scan_pool(
    pool_dir: pathlib.Path,
    ingested: dict,
) -> list[SyncItem]:
    """Scan the pool directory and classify each .md file.

    NEW       — filename not in state
    CHANGED   — filename in state but hash differs
    UNCHANGED — filename in state with matching hash (skip on sync)
    """
    items: list[SyncItem] = []
    for md_file in sorted(pool_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        h = sha256(text)
        prev = ingested.get(md_file.name)
        if prev is None:
            action = "NEW"
        elif prev["hash"] != h:
            action = "CHANGED"
        else:
            action = "UNCHANGED"
        items.append(SyncItem(md_file, text, h, action, title=extract_title(text)))
    return items


def sync(
    pool_dir: pathlib.Path,
    state_file: pathlib.Path,
    client: IngestClient,
    limit: int,
    dry_run: bool = False,
    only: str | None = None,
) -> SyncResult:
    """Run one sync pass. Idempotent - a second call right after the first
    finds all UNCHANGED and does nothing.
    """
    if not pool_dir.exists():
        raise FileNotFoundError(f"Pool dir missing: {pool_dir}")

    state = load_state(state_file)
    ingested = state.setdefault("ingested", {})
    tracked_before = len(ingested)

    all_items = scan_pool(pool_dir, ingested)
    todo = [i for i in all_items if i.action != "UNCHANGED"]
    if only:
        todo = [i for i in todo if i.md_file.name == only]

    log.info(
        "pool has %d docs, %d already tracked, %d pending",
        len(all_items), tracked_before, len(todo),
    )

    if not todo:
        return SyncResult(
            pool_total=len(all_items),
            tracked_before=tracked_before,
            done=0, failed=0, pending=0,
            tracked_after=tracked_before,
        )

    slice_ = todo[:limit] if limit > 0 else todo
    log.info("will process %d this run (limit=%d, dry_run=%s)",
             len(slice_), limit, dry_run)

    done, failed = 0, 0
    processed: list[SyncedDoc] = []
    total_start = time.perf_counter()

    for item in slice_:
        if dry_run:
            log.info("  DRY-RUN  %-9s %s (%d bytes)  title=%s",
                     item.action, item.md_file.name,
                     len(item.text.encode()), item.title[:60] if item.title else "-")
            continue
        item_start = time.perf_counter()
        try:
            resp = client.ingest(item.md_file, item.text)
            dur_ms = int((time.perf_counter() - item_start) * 1000)
            ingested[item.md_file.name] = {
                "hash": item.hash,
                "doc_id": resp["doc_id"],
                "chunks": resp["chunks"],
                "action": item.action,
                "title": item.title,
                "duration_ms": dur_ms,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            processed.append(SyncedDoc(
                filename=item.md_file.name,
                title=item.title,
                action=item.action,
                chunks=resp["chunks"],
                bytes=resp["bytes"],
                duration_ms=dur_ms,
            ))
            log.info(
                "  %-8s %s -> %d chunks (%d bytes) %dms  %s",
                item.action, item.md_file.name,
                resp["chunks"], resp["bytes"], dur_ms,
                item.title[:60] if item.title else "",
            )
            done += 1
        except Exception as e:  # noqa: BLE001
            dur_ms = int((time.perf_counter() - item_start) * 1000)
            log.error("  FAIL     %s: %s (after %dms)", item.md_file.name, e, dur_ms)
            failed += 1

    total_ms = int((time.perf_counter() - total_start) * 1000)

    if not dry_run:
        save_state(state_file, state)

    return SyncResult(
        pool_total=len(all_items),
        tracked_before=tracked_before,
        done=done,
        failed=failed,
        pending=len(todo) - done - failed,
        tracked_after=len(ingested),
        total_duration_ms=total_ms,
        processed=processed,
    )
