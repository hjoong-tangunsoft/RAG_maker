"""Extract structured metadata from Jira-exported markdown for filtered retrieval.

Anti-Hallucination Guard uses text-level defenses. Phase D uses STRUCTURED
metadata: parse the Jira fields from exporter output at ingest time so Chroma
can filter and sort by status/date instead of relying purely on cosine
similarity. Without this, an issue tracker with 83% completed issues will
always return completed issues to "무슨 일 하고 있어?" queries.

The exporter (scripts/exporters/jira_export.py) writes markdown like:

    # [MAN-1] 삼성 DS 데모 준비
    - **Type:** 작업
    - **Status:** 완료
    - **Priority:** High
    - **Assignee:** 진승민
    - **Created:** 2026-04-13
    - **Updated:** 2026-05-08

We parse those field lines and normalize status across Korean/English variants.
"""
from __future__ import annotations

import re
from datetime import datetime

# Jira issue key in the H1 heading (e.g., "# [MAN-1] ...")
_KEY_RE = re.compile(r"^#\s*\[(?P<key>[A-Z][A-Z0-9]*-\d+)\]", re.MULTILINE)

# "- **Field:** value" lines. Value can contain any character except newline.
_FIELD_RE = re.compile(
    r"^-\s*\*\*(?P<field>[^:*]+):\*\*\s*(?P<value>.+?)\s*$",
    re.MULTILINE,
)

# Status normalization: multiple Jira workflows use different labels for
# semantically identical states. Collapse to a small controlled vocabulary
# so retrieval filters can use one comparison across all issues.
_STATUS_COMPLETED = {"완료", "Done", "Closed", "Resolved", "닫힘", "종료"}
_STATUS_IN_PROGRESS = {"진행 중", "진행중", "In Progress", "In-Progress"}
_STATUS_TODO = {"해야 할 일", "해야할일", "To Do", "Open", "New", "TODO"}
_STATUS_ON_HOLD = {"보류", "On Hold", "Blocked", "Pending"}


def _normalize_status(raw: str) -> str:
    """Map a raw status label to a normalized bucket used by retrieval filters."""
    s = raw.strip()
    if s in _STATUS_COMPLETED:
        return "completed"
    if s in _STATUS_IN_PROGRESS:
        return "in_progress"
    if s in _STATUS_TODO:
        return "todo"
    if s in _STATUS_ON_HOLD:
        return "on_hold"
    return "other"


def _parse_date(s: str) -> int | None:
    """Convert 'YYYY-MM-DD' to unix timestamp (UTC). Return None on parse failure."""
    try:
        return int(datetime.strptime(s.strip(), "%Y-%m-%d").timestamp())
    except (ValueError, TypeError):
        return None


def parse_jira_metadata(text: str) -> dict[str, str | int] | None:
    """Detect Jira-format markdown and extract structured metadata.

    Returns None if the text is not a Jira export (no [PROJECT-N] heading).
    Otherwise returns a dict with a subset of:
        jira_key           str, e.g. "MAN-1"
        jira_status        str, one of: completed / in_progress / todo / on_hold / other
        jira_status_raw    str, original label (완료, In Progress, 등)
        jira_priority      str, e.g. "High"
        jira_assignee      str
        jira_type          str, e.g. "작업" / "버그"
        jira_created       str, "YYYY-MM-DD"
        jira_created_ts    int, unix timestamp for date sorting
        jira_updated       str, "YYYY-MM-DD"
        jira_updated_ts    int, unix timestamp

    Non-Jira documents (policy notes, onboarding checklists) return None so
    they are stored without jira_* fields and never picked up by the
    active-work status filter.

    All values are primitives (str/int) to satisfy Chroma's metadata schema.
    """
    key_match = _KEY_RE.search(text)
    if not key_match:
        return None

    fields: dict[str, str] = {}
    for m in _FIELD_RE.finditer(text):
        fields[m.group("field").strip().lower()] = m.group("value").strip()

    meta: dict[str, str | int] = {"jira_key": key_match.group("key")}

    if raw_status := fields.get("status"):
        meta["jira_status_raw"] = raw_status
        meta["jira_status"] = _normalize_status(raw_status)

    if priority := fields.get("priority"):
        # Some issues have "(없음)" placeholder; keep only real priorities.
        if priority and priority != "(없음)":
            meta["jira_priority"] = priority

    if assignee := fields.get("assignee"):
        meta["jira_assignee"] = assignee

    if issuetype := fields.get("type"):
        meta["jira_type"] = issuetype

    if created := fields.get("created"):
        if ts := _parse_date(created):
            meta["jira_created"] = created
            meta["jira_created_ts"] = ts

    if updated := fields.get("updated"):
        if ts := _parse_date(updated):
            meta["jira_updated"] = updated
            meta["jira_updated_ts"] = ts

    return meta
