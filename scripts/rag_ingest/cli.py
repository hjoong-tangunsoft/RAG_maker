"""CLI entry point for rag-ingest.

Subcommands:
    sync      Ingest pending documents from pool
    status    Show pool + state summary (source of truth: filesystem + state.json)
    ls        List documents currently in RAG server (source of truth: server)
    refresh   Force re-ingest a specific document
    reset     Clear state file (all pool docs will re-ingest next sync)
    health    Check RAG server health

Global options apply to all subcommands (pool dir, state file, server URL).
Environment variable defaults match scripts/rag_sync.py exactly - so switching
between the raw script and this CLI is transparent.
"""
from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

from . import __version__
from .core import scan_pool, sync
from .ingest_client import IngestClient
from .state import load_state, save_state

# Defaults (env override, matching rag_sync.py exactly)
DEFAULT_POOL_DIR = pathlib.Path(
    os.environ.get("POOL_DIR", "/upload/rag/data/docs-pool")
)
DEFAULT_STATE_FILE = pathlib.Path(
    os.environ.get("STATE_FILE", "/upload/rag/data/sync-state.json")
)
DEFAULT_RAG_URL = os.environ.get("RAG_URL", "http://127.0.0.1:8100")
DEFAULT_DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "2"))


def setup_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
    )


# --------- subcommand handlers ---------

def cmd_sync(args) -> int:
    client = IngestClient(args.rag_url)
    try:
        result = sync(
            pool_dir=args.pool_dir,
            state_file=args.state_file,
            client=client,
            limit=args.limit,
            dry_run=args.dry_run,
            only=args.only,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    prefix = "[DRY-RUN] " if args.dry_run else ""
    print(f"\n{prefix}Sync complete:")
    print(f"  Pool total:      {result.pool_total}")
    print(f"  Tracked before:  {result.tracked_before}")
    print(f"  Done this run:   {result.done}")
    print(f"  Failed:          {result.failed}")
    print(f"  Still pending:   {result.pending}")
    print(f"  Tracked after:   {result.tracked_after}")
    return 0 if result.failed == 0 else 2


def cmd_status(args) -> int:
    if not args.pool_dir.exists():
        print(f"Pool dir missing: {args.pool_dir}", file=sys.stderr)
        return 1

    state = load_state(args.state_file)
    ingested = state.get("ingested", {})
    items = scan_pool(args.pool_dir, ingested)

    print(f"Pool location:   {args.pool_dir}")
    print(f"State file:      {args.state_file}")
    print(f"Server:          {args.rag_url}")
    print()
    print(f"{'DOCUMENT':<40} {'HASH':<10} {'STATUS':<10} INGESTED")
    print(f"{'-'*40} {'-'*10} {'-'*10} {'-'*24}")
    marker = {"NEW": "○ NEW", "CHANGED": "△ CHG", "UNCHANGED": "● OK"}
    for it in items:
        h_short = it.hash[:8]
        prev = ingested.get(it.md_file.name, {})
        ingested_at = (prev.get("ingested_at") or "-")[:19]
        print(f"{it.md_file.name:<40} {h_short:<10} {marker[it.action]:<10} {ingested_at}")
    print()
    counts = {a: sum(1 for i in items if i.action == a)
              for a in ("NEW", "CHANGED", "UNCHANGED")}
    pending = counts["NEW"] + counts["CHANGED"]
    print(f"Summary: {len(items)} total, {counts['UNCHANGED']} up-to-date, "
          f"{pending} pending ({counts['NEW']} NEW + {counts['CHANGED']} CHANGED)")
    return 0


def cmd_ls(args) -> int:
    client = IngestClient(args.rag_url)
    try:
        docs = client.list_docs()
    except Exception as e:  # noqa: BLE001
        print(f"Server error: {e}", file=sys.stderr)
        return 1
    if not docs:
        print("(no documents in RAG)")
        return 0
    print(f"{'DOC_ID':<38} {'SOURCE':<36} {'CHUNKS':>6}  ADDED")
    print(f"{'-'*38} {'-'*36} {'-'*6}  {'-'*20}")
    for d in docs:
        did = d["doc_id"][:36]
        src = d["source"][:34]
        print(f"{did:<38} {src:<36} {d['chunks']:>6}  {d['added_at'][:19]}")
    print(f"\nTotal: {len(docs)} documents")
    return 0


def cmd_refresh(args) -> int:
    state = load_state(args.state_file)
    ingested = state.get("ingested", {})
    filename = args.filename
    if filename not in ingested:
        print(f"Error: '{filename}' not tracked in state.", file=sys.stderr)
        tracked = list(ingested.keys())
        preview = ", ".join(tracked[:5]) + ("..." if len(tracked) > 5 else "")
        print(f"Currently tracked: {preview}", file=sys.stderr)
        return 1
    del ingested[filename]
    save_state(args.state_file, state)
    print(f"Hash cleared for '{filename}'. Next sync will re-ingest.")
    if args.now:
        print("Running sync now (--now flag)...")
        client = IngestClient(args.rag_url)
        try:
            result = sync(
                pool_dir=args.pool_dir,
                state_file=args.state_file,
                client=client,
                limit=1,
                dry_run=False,
                only=filename,
            )
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Result: {result.done} done, {result.failed} failed")
        return 0 if result.failed == 0 else 2
    return 0


def cmd_reset(args) -> int:
    if not args.confirm:
        print("This will clear the ENTIRE state file. On next sync, all pool",
              file=sys.stderr)
        print("documents will be re-ingested (server overwrites existing chunks).",
              file=sys.stderr)
        print("Add --confirm to proceed.", file=sys.stderr)
        return 1
    if args.state_file.exists():
        args.state_file.unlink()
        print(f"Deleted state file: {args.state_file}")
    else:
        print(f"Nothing to delete (state file does not exist): {args.state_file}")
    return 0


def cmd_health(args) -> int:
    client = IngestClient(args.rag_url)
    try:
        h = client.health()
    except Exception as e:  # noqa: BLE001
        print(f"Server unreachable at {args.rag_url}: {e}", file=sys.stderr)
        return 1
    print(f"Server:      {client.rag_root}")
    print(f"Status:      {h.get('status')}")
    print(f"Upstream:    {h.get('upstream_llm')}")
    print(f"Embed model: {h.get('embed_model')}")
    print(f"Embed dim:   {h.get('embed_dim')}")
    return 0


# --------- parser ---------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag-ingest",
        description="RAG document ingest pipeline (Dumb Server, Smart Client).",
        epilog="Docs: https://github.com/hjoong-tangunsoft/RAG_maker/blob/RAG_maker/docs/DESIGN_PRINCIPLES.md",
    )
    p.add_argument("--version", action="version",
                   version=f"rag-ingest {__version__}")
    p.add_argument("-v", "--verbose", action="count", default=1,
                   help="Increase verbosity (-v info, -vv debug)")
    p.add_argument("--pool-dir", type=pathlib.Path, default=DEFAULT_POOL_DIR,
                   metavar="PATH",
                   help=f"Directory to scan for .md files "
                        f"(default: {DEFAULT_POOL_DIR})")
    p.add_argument("--state-file", type=pathlib.Path, default=DEFAULT_STATE_FILE,
                   metavar="PATH",
                   help=f"State JSON file path "
                        f"(default: {DEFAULT_STATE_FILE})")
    p.add_argument("--rag-url", default=DEFAULT_RAG_URL, metavar="URL",
                   help=f"RAG server base URL "
                        f"(default: {DEFAULT_RAG_URL})")

    sub = p.add_subparsers(dest="command", required=True,
                           metavar="COMMAND")

    # sync
    s = sub.add_parser("sync",
                       help="Ingest pending documents from pool")
    s.add_argument("--limit", type=int, default=DEFAULT_DAILY_LIMIT,
                   help=f"Max ingests this run "
                        f"(default: {DEFAULT_DAILY_LIMIT}, 0=no limit)")
    s.add_argument("--dry-run", action="store_true",
                   help="Show what would happen without calling server")
    s.add_argument("--only", metavar="FILENAME",
                   help="Only process this specific filename")
    s.set_defaults(func=cmd_sync)

    # status
    st = sub.add_parser("status",
                        help="Show pool + state summary (local view)")
    st.set_defaults(func=cmd_status)

    # ls
    l = sub.add_parser("ls",
                       help="List docs currently in RAG server (server view)")
    l.set_defaults(func=cmd_ls)

    # refresh
    r = sub.add_parser("refresh",
                       help="Force re-ingest a specific document")
    r.add_argument("filename", help="Filename as it appears in pool")
    r.add_argument("--now", action="store_true",
                   help="Run sync immediately after clearing hash")
    r.set_defaults(func=cmd_refresh)

    # reset
    rs = sub.add_parser("reset",
                        help="Clear state file (all docs will re-ingest)")
    rs.add_argument("--confirm", action="store_true",
                    help="Actually do the deletion")
    rs.set_defaults(func=cmd_reset)

    # health
    h = sub.add_parser("health",
                       help="Check RAG server health")
    h.set_defaults(func=cmd_health)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
