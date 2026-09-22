#!/usr/bin/env python3
"""
RAG evaluation harness - end-to-end quality regression tests.

Runs the golden set (evals/golden_set.jsonl) against a deployed RAG service
and produces per-entry + per-category metrics.

Categories (from Issue #8):
  factual      - 문서에 있는 정보 (Hit@1, Hit@3, keyword coverage)
  summary      - 다중 청크 종합
  multi_hop    - 여러 문서 종합 (Recall@5)
  absence      - 문서에 없어야 "모른다" 답변 (absence_accuracy)
  adversarial  - 프롬프트 인젝션 저항 (must_not_contain)

Metrics:
  Hit@1, Hit@3   - top-k 안에 expected_docs 중 하나 이상 있으면 pass
  Recall@5       - top-5 안에 expected_docs 중 몇 % 포함 (multi-doc용)
  MRR            - Mean Reciprocal Rank of first expected doc
  keyword%       - answer 에 expected_keywords 몇 % 포함
  absence acc    - should_say_unknown=true 일 때 answer 가 "자료에 없" 패턴 매칭
  injection rst  - adversarial 쿼리에 대해 must_not_contain 준수 (인젝션 거부)

Usage:
    BASE=http://127.0.0.1:8100 python3 scripts/eval_rag.py
    BASE=https://llm.tangunsoft.com API_KEY=xxx python3 scripts/eval_rag.py

    python3 scripts/eval_rag.py --category factual       # single category
    python3 scripts/eval_rag.py --save-run               # timestamped JSONL log
    python3 scripts/eval_rag.py --fail-below 0.7         # exit 1 if Hit@1 < 70%

Exit codes:
    0 = all thresholds passed (or --fail-below not set)
    1 = failed threshold check
    2 = infrastructure error (network, malformed golden set, etc.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import httpx

# Windows cp949 default terminal can't print non-BMP chars (emoji, etc.).
# Force UTF-8 stdout so the golden set's Korean + emoji queries print safely.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


BASE = os.environ.get("BASE", "http://127.0.0.1:8100").rstrip("/")
API_KEY = os.environ.get("API_KEY")

# Repo-root relative paths (works from repo root or from scripts/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SET_PATH = _REPO_ROOT / "evals" / "golden_set.jsonl"
OUTPUT_DIR = _REPO_ROOT / "eval-runs"

# Patterns the LLM emits when refusing to answer (absence detection)
ABSENCE_PATTERNS = [
    "자료에 없",
    "정보에 없",
    "확인되지 않",
    "관련 자료가 없",
    "명확히 찾지 못했",
    "찾을 수 없",
    "없습니다",
]


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _client() -> httpx.Client:
    return httpx.Client(timeout=180.0, verify=False, headers=_headers())


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"ERROR: golden set not found at {path}", file=sys.stderr)
        sys.exit(2)
    entries = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR line {line_no}: {e}", file=sys.stderr)
                sys.exit(2)
    return entries


@dataclass
class EvalResult:
    id: str
    category: str
    query: str
    expected_docs: list[str]
    top_docs: list[tuple[str, float]]
    answer_preview: str
    hit1: bool
    hit3: bool
    recall5: float
    mrr: float
    keyword_coverage: float
    absence_correct: bool | None = None      # None = not applicable
    injection_resisted: bool | None = None    # None = not applicable
    error: str | None = None


def run_query(client: httpx.Client, entry: dict[str, Any]) -> EvalResult:
    query = entry["query"]
    expected: list[str] = entry.get("expected_docs", []) or []
    keywords: list[str] = entry.get("expected_keywords", []) or []
    should_say_unknown = bool(entry.get("should_say_unknown", False))
    should_maintain_rules = bool(entry.get("should_maintain_rules", False))
    must_not_contain: list[str] = entry.get("must_not_contain", []) or []

    # Empty defaults
    top_docs: list[tuple[str, float]] = []
    answer = ""

    try:
        r = client.post(f"{BASE}/rag/search", json={"query": query, "k": 5})
        r.raise_for_status()
        hits = r.json().get("hits", [])
        top_docs = [(h["doc_id"], float(h["score"])) for h in hits]

        r2 = client.post(
            f"{BASE}/rag/query",
            json={"query": query, "k": 5, "max_tokens": 300},
        )
        r2.raise_for_status()
        answer = r2.json().get("answer", "")
    except Exception as e:  # noqa: BLE001
        return EvalResult(
            id=entry["id"],
            category=entry["category"],
            query=query,
            expected_docs=expected,
            top_docs=[],
            answer_preview="",
            hit1=False, hit3=False, recall5=0.0, mrr=0.0, keyword_coverage=0.0,
            error=str(e)[:200],
        )

    # ---- Retrieval metrics ----
    top_ids = [d for d, _ in top_docs]
    expected_set = set(expected)

    hit1 = bool(expected_set & set(top_ids[:1])) if expected else False
    hit3 = bool(expected_set & set(top_ids[:3])) if expected else False
    recall5 = (
        len(expected_set & set(top_ids[:5])) / len(expected_set)
        if expected else 0.0
    )
    mrr = 0.0
    if expected:
        for i, doc in enumerate(top_ids, 1):
            if doc in expected_set:
                mrr = 1.0 / i
                break

    # ---- Keyword coverage (generation faithfulness proxy) ----
    keyword_coverage = 0.0
    if keywords:
        keyword_coverage = sum(1 for kw in keywords if kw in answer) / len(keywords)

    # ---- Absence detection ----
    absence_correct: bool | None = None
    if should_say_unknown:
        absence_correct = any(p in answer for p in ABSENCE_PATTERNS)

    # ---- Injection resistance ----
    injection_resisted: bool | None = None
    if should_maintain_rules:
        # Two conditions:
        # 1. must_not_contain patterns absent from answer
        # 2. answer isn't empty (LLM didn't just crash)
        no_forbidden = all(m not in answer for m in must_not_contain)
        has_content = len(answer.strip()) > 10
        injection_resisted = no_forbidden and has_content

    return EvalResult(
        id=entry["id"],
        category=entry["category"],
        query=query,
        expected_docs=expected,
        top_docs=top_docs[:5],
        answer_preview=answer[:300],
        hit1=hit1,
        hit3=hit3,
        recall5=recall5,
        mrr=mrr,
        keyword_coverage=keyword_coverage,
        absence_correct=absence_correct,
        injection_resisted=injection_resisted,
    )


def aggregate(results: list[EvalResult]) -> dict[str, Any]:
    by_cat: dict[str, list[EvalResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    cat_metrics: dict[str, dict[str, float | int]] = {}
    for cat, rs in by_cat.items():
        n = len(rs)
        with_expected = [r for r in rs if r.expected_docs]
        m = with_expected or rs
        cat_metrics[cat] = {
            "count": n,
            "hit1_rate": (sum(r.hit1 for r in with_expected) / len(with_expected)) if with_expected else 0.0,
            "hit3_rate": (sum(r.hit3 for r in with_expected) / len(with_expected)) if with_expected else 0.0,
            "avg_recall5": (sum(r.recall5 for r in with_expected) / len(with_expected)) if with_expected else 0.0,
            "avg_mrr": (sum(r.mrr for r in with_expected) / len(with_expected)) if with_expected else 0.0,
            "avg_keyword_coverage": sum(r.keyword_coverage for r in rs) / n if n else 0.0,
        }
        abs_rs = [r for r in rs if r.absence_correct is not None]
        if abs_rs:
            cat_metrics[cat]["absence_accuracy"] = sum(bool(r.absence_correct) for r in abs_rs) / len(abs_rs)
        inj_rs = [r for r in rs if r.injection_resisted is not None]
        if inj_rs:
            cat_metrics[cat]["injection_resistance"] = sum(bool(r.injection_resisted) for r in inj_rs) / len(inj_rs)

    # Overall
    with_expected = [r for r in results if r.expected_docs]
    overall: dict[str, float | int] = {
        "total": len(results),
        "errors": sum(1 for r in results if r.error),
        "hit1_rate": (sum(r.hit1 for r in with_expected) / len(with_expected)) if with_expected else 0.0,
        "hit3_rate": (sum(r.hit3 for r in with_expected) / len(with_expected)) if with_expected else 0.0,
        "avg_mrr": (sum(r.mrr for r in with_expected) / len(with_expected)) if with_expected else 0.0,
    }
    abs_rs = [r for r in results if r.absence_correct is not None]
    if abs_rs:
        overall["absence_accuracy"] = sum(bool(r.absence_correct) for r in abs_rs) / len(abs_rs)
    inj_rs = [r for r in results if r.injection_resisted is not None]
    if inj_rs:
        overall["injection_resistance"] = sum(bool(r.injection_resisted) for r in inj_rs) / len(inj_rs)

    return {
        "timestamp": int(time.time()),
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": BASE,
        "overall": overall,
        "by_category": cat_metrics,
    }


def _fmt_bool(b: bool | None) -> str:
    if b is None:
        return "-"
    return "PASS" if b else "FAIL"


def print_report(results: list[EvalResult], summary: dict[str, Any]) -> None:
    print()
    print("=" * 100)
    print(f"RAG Eval Results   BASE={BASE}   iso_time={summary['iso_time']}")
    print("=" * 100)
    print()

    # Per-entry table
    header = (
        f"  {'id':<6} {'category':<12} {'H@1':<5} {'H@3':<5} "
        f"{'recall':<7} {'mrr':<5} {'kw%':<5} {'abs':<5} {'inj':<5} query"
    )
    print(header)
    print(f"  {'-'*6} {'-'*12} {'-'*5} {'-'*5} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*5} " + "-" * 40)
    for r in results:
        if r.error:
            print(f"  {r.id:<6} {r.category:<12} ERR  {r.error[:50]}")
            continue
        h1 = "PASS" if r.hit1 else "fail"
        h3 = "PASS" if r.hit3 else "fail"
        abs_s = _fmt_bool(r.absence_correct)
        inj_s = _fmt_bool(r.injection_resisted)
        q_short = r.query if len(r.query) < 40 else r.query[:37] + "..."
        # For entries without expected_docs, H@1/H@3 don't apply
        if not r.expected_docs:
            h1 = "-"
            h3 = "-"
        print(
            f"  {r.id:<6} {r.category:<12} {h1:<5} {h3:<5} "
            f"{r.recall5:<7.2f} {r.mrr:<5.2f} {r.keyword_coverage*100:<5.0f} "
            f"{abs_s:<5} {inj_s:<5} {q_short}"
        )

    # Category summary
    print()
    print("=" * 100)
    print("SUMMARY BY CATEGORY")
    print("=" * 100)
    for cat, m in summary["by_category"].items():
        print(f"\n  [{cat}]   ({int(m['count'])} queries)")
        if m.get("hit1_rate", 0) or "hit1_rate" in m:
            print(f"    Hit@1:              {m['hit1_rate']*100:5.0f}%")
            print(f"    Hit@3:              {m['hit3_rate']*100:5.0f}%")
            print(f"    Avg Recall@5:       {m['avg_recall5']:5.2f}")
            print(f"    Avg MRR:            {m['avg_mrr']:5.2f}")
        if m['avg_keyword_coverage'] > 0:
            print(f"    Avg Keyword%:       {m['avg_keyword_coverage']*100:5.0f}%")
        if "absence_accuracy" in m:
            print(f"    Absence Accuracy:   {m['absence_accuracy']*100:5.0f}%")
        if "injection_resistance" in m:
            print(f"    Injection Resist:   {m['injection_resistance']*100:5.0f}%")

    print()
    print("=" * 100)
    print("OVERALL")
    print("=" * 100)
    o = summary["overall"]
    print(f"  Total queries:        {int(o['total'])}")
    print(f"  Errors:               {int(o['errors'])}")
    print(f"  Hit@1:                {o['hit1_rate']*100:5.0f}%")
    print(f"  Hit@3:                {o['hit3_rate']*100:5.0f}%")
    print(f"  Avg MRR:              {o['avg_mrr']:5.2f}")
    if "absence_accuracy" in o:
        print(f"  Absence Accuracy:     {o['absence_accuracy']*100:5.0f}%")
    if "injection_resistance" in o:
        print(f"  Injection Resistance: {o['injection_resistance']*100:5.0f}%")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG eval harness (Issue #8)")
    ap.add_argument(
        "--category",
        choices=["factual", "summary", "multi_hop", "absence", "adversarial"],
        default=None,
        help="run only this category",
    )
    ap.add_argument("--output", type=Path, default=None, help="save full JSON report")
    ap.add_argument(
        "--save-run",
        action="store_true",
        help=f"save timestamped JSONL to {OUTPUT_DIR}/",
    )
    ap.add_argument(
        "--fail-below",
        type=float,
        default=None,
        help="exit 1 if overall Hit@1 rate falls below this (0.0-1.0)",
    )
    ap.add_argument(
        "--golden-set",
        type=Path,
        default=GOLDEN_SET_PATH,
        help="path to golden set JSONL",
    )
    args = ap.parse_args()

    entries = load_golden_set(args.golden_set)
    if args.category:
        entries = [e for e in entries if e["category"] == args.category]
    if not entries:
        print("No entries to evaluate.", file=sys.stderr)
        return 2

    print(f"Loaded {len(entries)} entries from {args.golden_set}")
    print(f"BASE: {BASE}")
    print()

    results: list[EvalResult] = []
    with _client() as client:
        try:
            r = client.get(f"{BASE}/rag/health")
            r.raise_for_status()
            print(f"Health: {r.json()}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: health check failed: {e}", file=sys.stderr)
            return 2

        print()
        for i, e in enumerate(entries, 1):
            print(f"  [{i:>2}/{len(entries)}] {e['id']} ({e['category']}) - {e['query'][:60]}")
            results.append(run_query(client, e))

    summary = aggregate(results)
    print_report(results, summary)

    if args.output:
        args.output.write_text(
            json.dumps(
                {"summary": summary, "results": [asdict(r) for r in results]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Full report saved to {args.output}")

    if args.save_run:
        OUTPUT_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_path = OUTPUT_DIR / f"run-{ts}.jsonl"
        with run_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"__summary__": summary}, ensure_ascii=False) + "\n")
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        print(f"Regression run saved to {run_path}")

    if args.fail_below is not None:
        rate = summary["overall"]["hit1_rate"]
        if rate < args.fail_below:
            print(
                f"\nFAIL: overall Hit@1 {rate:.2%} < threshold {args.fail_below:.0%}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
