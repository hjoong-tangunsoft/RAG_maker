"""Core sync logic. Same behavior as rag_sync.py but as a reusable function.

Every CLI subcommand that ingests calls `sync()` here. The only difference
from rag_sync.py is packaging: same NEW/CHANGED/UNCHANGED classification,
same hash-based dedup, same idempotency guarantees.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone

from .ingest_client import IngestClient
from .state import load_state, save_state, sha256

log = logging.getLogger("rag-ingest.core")


@dataclass
class SyncItem:
    md_file: pathlib.Path
    text: str
    hash: str
    action: str  # "NEW" | "CHANGED" | "UNCHANGED"


@dataclass
class SyncResult:
    pool_total: int
    tracked_before: int
    done: int
    failed: int
    pending: int
    tracked_after: int


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
        items.append(SyncItem(md_file, text, h, action))
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
    for item in slice_:
        if dry_run:
            log.info("  DRY-RUN  %-9s %s (%d bytes)",
                     item.action, item.md_file.name, len(item.text.encode()))
            continue
        try:
            resp = client.ingest(item.md_file, item.text)
            ingested[item.md_file.name] = {
                "hash": item.hash,
                "doc_id": resp["doc_id"],
                "chunks": resp["chunks"],
                "action": item.action,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            log.info(
                "  %-8s %s -> %d chunks (%d bytes)",
                item.action, item.md_file.name,
                resp["chunks"], resp["bytes"],
            )
            done += 1
        except Exception as e:  # noqa: BLE001
            log.error("  FAIL     %s: %s", item.md_file.name, e)
            failed += 1

    if not dry_run:
        save_state(state_file, state)

    return SyncResult(
        pool_total=len(all_items),
        tracked_before=tracked_before,
        done=done,
        failed=failed,
        pending=len(todo) - done - failed,
        tracked_after=len(ingested),
    )
