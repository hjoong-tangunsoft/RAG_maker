#!/usr/bin/env python3
"""Jira issue exporter → RAG document pool.

Fetches issues from an Atlassian Cloud Jira project via the new
/rest/api/3/search/jql endpoint (migrated from deprecated /search) and
writes each as a markdown file to the pool directory. The existing
`rag-ingest sync` (Smart Client) then picks them up 2/day.

Follows the "Dumb Server, Smart Client" principle from
docs/DESIGN_PRINCIPLES.md — this exporter is one such client for the
Jira source. Same code path for Notion/Confluence exporters when added.

Environment variables:
    JIRA_URL       Base URL, e.g. https://xxx.atlassian.net
    JIRA_EMAIL     Email for API auth
    JIRA_TOKEN     API token (get from id.atlassian.com/manage-profile/security/api-tokens)
    JIRA_PROJECT   Project key (e.g. MAN)
    JIRA_JQL       Optional JQL suffix (default: ORDER BY updated DESC)
    POOL_DIR       Where to write .md files (default: /upload/rag/data/docs-pool)

Idempotent: files already matching current content are skipped.
"""
from __future__ import annotations

import argparse
import base64
import html.parser
import logging
import os
import pathlib
import re
import sys
import time
from typing import Any

import httpx

# ---------- config ----------

JIRA_URL = os.environ.get("JIRA_URL", "").rstrip("/")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "")
JIRA_TOKEN = os.environ.get("JIRA_TOKEN", "")
JIRA_PROJECT = os.environ.get("JIRA_PROJECT", "MAN")
JIRA_JQL_SUFFIX = os.environ.get("JIRA_JQL", "ORDER BY updated DESC")
POOL_DIR = pathlib.Path(
    os.environ.get("POOL_DIR", "/upload/rag/data/docs-pool")
)

log = logging.getLogger("jira-export")


# ---------- HTML → text ----------

class _HTMLToMarkdown(html.parser.HTMLParser):
    """Minimal HTML → markdown-ish converter for Jira renderedFields."""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._href: str | None = None
        self._link_start_idx: int | None = None

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag in ("p", "div"):
            pass
        elif tag == "br":
            self.parts.append("\n")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n" + "#" * int(tag[1]) + " ")
        elif tag == "code":
            self.parts.append("`")
        elif tag == "pre":
            self.parts.append("\n```\n")
        elif tag in ("strong", "b"):
            self.parts.append("**")
        elif tag in ("em", "i"):
            self.parts.append("_")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "a":
            self._href = d.get("href", "")
            self._link_start_idx = len(self.parts)
        elif tag == "img":
            alt = d.get("alt", "image")
            src = d.get("src", "")
            self.parts.append(f"![{alt}]({src})")

    def handle_endtag(self, tag):
        if tag in ("p", "div"):
            self.parts.append("\n\n")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n\n")
        elif tag == "code":
            self.parts.append("`")
        elif tag == "pre":
            self.parts.append("\n```\n")
        elif tag in ("strong", "b"):
            self.parts.append("**")
        elif tag in ("em", "i"):
            self.parts.append("_")
        elif tag == "a":
            if self._href and self._link_start_idx is not None:
                # Wrap accumulated text in [text](href)
                inner = "".join(self.parts[self._link_start_idx:])
                self.parts = self.parts[:self._link_start_idx]
                self.parts.append(f"[{inner}]({self._href})")
            self._href = None
            self._link_start_idx = None

    def handle_data(self, data):
        self.parts.append(data)

    def get_text(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


def html_to_markdown(html_str: str | None) -> str:
    if not html_str:
        return ""
    p = _HTMLToMarkdown()
    try:
        p.feed(html_str)
    except Exception:  # noqa: BLE001
        return html_str  # fall back to raw
    return p.get_text()


# ---------- Jira client ----------

def _auth_header(email: str, token: str) -> str:
    raw = f"{email}:{token}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def fetch_all_issues(client: httpx.Client, project: str, jql_suffix: str) -> list[dict]:
    """Fetch all matching issues via /rest/api/3/search/jql (paginated).

    Uses the new endpoint (old /rest/api/3/search deprecated 2025).
    """
    jql = f"project = {project} {jql_suffix}"
    all_issues: list[dict] = []
    next_token: str | None = None

    while True:
        params = {
            "jql": jql,
            "maxResults": 100,
            "fields": "summary,status,issuetype,assignee,reporter,labels,"
                      "created,updated,duedate,priority,description,comment",
            "expand": "renderedFields",
        }
        if next_token:
            params["nextPageToken"] = next_token

        r = client.get(f"{JIRA_URL}/rest/api/3/search/jql", params=params)
        r.raise_for_status()
        data = r.json()

        issues = data.get("issues", [])
        all_issues.extend(issues)
        log.debug("  page: got %d (total so far %d)", len(issues), len(all_issues))

        if data.get("isLast", True):
            break
        next_token = data.get("nextPageToken")
        if not next_token:
            break

    return all_issues


def _person(field: dict | None) -> str:
    if not field:
        return "미할당"
    return field.get("displayName") or field.get("emailAddress") or "?"


def format_issue(issue: dict) -> tuple[str, str]:
    """Convert a Jira issue dict → (padded_key, markdown_content)."""
    key = issue["key"]
    project, num = key.split("-", 1)
    padded = f"{project}-{int(num):04d}"

    fields = issue["fields"]
    rendered = issue.get("renderedFields", {}) or {}

    summary = fields.get("summary", "").strip()
    status = fields.get("status", {}).get("name", "")
    issuetype = fields.get("issuetype", {}).get("name", "")
    assignee = _person(fields.get("assignee"))
    reporter = _person(fields.get("reporter"))
    priority = (fields.get("priority") or {}).get("name", "")
    labels_list = fields.get("labels") or []
    labels = ", ".join(labels_list) if labels_list else "(없음)"
    created = fields.get("created", "")[:10]
    updated = fields.get("updated", "")[:10]
    duedate = fields.get("duedate") or ""
    url = f"{JIRA_URL}/browse/{key}"

    desc_md = html_to_markdown(rendered.get("description"))

    comments_meta = fields.get("comment", {}) or {}
    comments = comments_meta.get("comments", []) or []
    rendered_comments = (rendered.get("comment") or {}).get("comments", []) or []
    # Zip original + rendered (they have the same order)
    comments_paired: list[tuple[dict, str]] = []
    for i, cmt in enumerate(comments):
        body_html = ""
        if i < len(rendered_comments):
            body_html = rendered_comments[i].get("body", "")
        comments_paired.append((cmt, html_to_markdown(body_html)))

    # Assemble markdown
    lines = [
        f"# [{key}] {summary}",
        "",
        f"- **Type:** {issuetype}",
        f"- **Status:** {status}",
        f"- **Priority:** {priority or '(없음)'}",
        f"- **Assignee:** {assignee}",
        f"- **Reporter:** {reporter}",
        f"- **Labels:** {labels}",
        f"- **Created:** {created}",
        f"- **Updated:** {updated}",
    ]
    if duedate:
        lines.append(f"- **Due:** {duedate}")
    lines.extend([f"- **URL:** {url}", ""])

    lines.extend(["## Description", ""])
    lines.append(desc_md if desc_md else "(설명 없음)")

    if comments_paired:
        lines.extend(["", f"## Comments ({len(comments_paired)}건)", ""])
        for cmt, body_md in comments_paired:
            author = _person(cmt.get("author"))
            created_c = (cmt.get("created") or "")[:10]
            lines.append(f"### {author} @ {created_c}")
            lines.append("")
            lines.append(body_md or "(비어있음)")
            lines.append("")

    return padded, "\n".join(lines).rstrip() + "\n"


# ---------- main ----------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Jira issues from a project to markdown pool.",
        epilog="Idempotent — files with matching content are skipped.",
    )
    parser.add_argument("--project", default=JIRA_PROJECT,
                        help=f"Project key (default: {JIRA_PROJECT})")
    parser.add_argument("--jql", default=JIRA_JQL_SUFFIX,
                        help=f"JQL suffix (default: '{JIRA_JQL_SUFFIX}')")
    parser.add_argument("--pool-dir", type=pathlib.Path, default=POOL_DIR,
                        help=f"Output directory (default: {POOL_DIR})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be written without touching disk")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
    )

    if not JIRA_URL or not JIRA_EMAIL or not JIRA_TOKEN:
        log.error("JIRA_URL, JIRA_EMAIL, JIRA_TOKEN must be set")
        return 1

    args.pool_dir.mkdir(parents=True, exist_ok=True)
    log.info("Exporting from %s project=%s → %s", JIRA_URL, args.project, args.pool_dir)

    headers = {
        "Authorization": _auth_header(JIRA_EMAIL, JIRA_TOKEN),
        "Accept": "application/json",
    }
    total_start = time.perf_counter()

    with httpx.Client(headers=headers, timeout=60.0) as client:
        fetch_start = time.perf_counter()
        issues = fetch_all_issues(client, args.project, args.jql)
        fetch_ms = int((time.perf_counter() - fetch_start) * 1000)
        log.info("Fetched %d issues in %d ms", len(issues), fetch_ms)

    written = 0
    unchanged = 0
    failed = 0
    for issue in issues:
        try:
            padded, content = format_issue(issue)
        except Exception as e:  # noqa: BLE001
            log.error("  FAIL     %s: %s", issue.get("key", "?"), e)
            failed += 1
            continue

        filename = f"jira-{padded}.md"
        path = args.pool_dir / filename
        size_bytes = len(content.encode("utf-8"))

        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing == content:
                unchanged += 1
                log.debug("  =        %s (%d bytes, unchanged)", filename, size_bytes)
                continue

        if args.dry_run:
            log.info("  [DRY]    %s (%d bytes)", filename, size_bytes)
            continue

        path.write_text(content, encoding="utf-8")
        log.info("  ✓        %s (%d bytes)", filename, size_bytes)
        written += 1

    total_ms = int((time.perf_counter() - total_start) * 1000)
    log.info("Export complete: %d written, %d unchanged, %d failed, %d ms total",
             written, unchanged, failed, total_ms)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
