#!/usr/bin/env python3
"""
Embedding & retrieval quality evaluation - runs entirely against the deployed
RAG service so it works both on the server (localhost) and from any client
once :443 is reachable.

  PART A - Direct embedding similarity via POST /rag/debug/similarity
    Pairs of texts are embedded and compared with cosine similarity.
    Verifies the raw embedding model produces semantically meaningful
    scores independent of any indexing.

  PART B - End-to-end retrieval via /rag/ingest + /rag/search
    Ingests a labeled corpus and runs a suite of queries with known
    expected top-1 doc_id. Measures Hit@1, Hit@3, cross-lingual accuracy,
    and score gap between correct and incorrect hits.

Usage
-----
    BASE=http://127.0.0.1:8100 python3 eval_embedding.py
    BASE=https://llm.tangunsoft.com API_KEY=xxx python3 eval_embedding.py
    python3 eval_embedding.py --part a    # skip retrieval

Exit 0 if all thresholds pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import httpx


BASE = os.environ.get("BASE", "http://127.0.0.1:8100").rstrip("/")
API_KEY = os.environ.get("API_KEY")
TEST_DOC_PREFIX = "eval-"


CORPUS: dict[str, tuple[str, str, str, str]] = {
    f"{TEST_DOC_PREFIX}ko-company": (
        "tangun-intro.md",
        "탄군소프트(TangunSoft)는 서울에 본사를 둔 한국의 소프트웨어 개발 회사입니다. "
        "주요 사업은 클라우드 인프라 컨설팅, GitHub Enterprise Server 도입, 사내 LLM 시스템 구축입니다. "
        "2020년 설립되었으며 대표 제품은 사내 코드 어시스턴트입니다.",
        "ko", "company",
    ),
    f"{TEST_DOC_PREFIX}ko-recipe": (
        "kimchi-jjigae.md",
        "김치찌개는 잘 익은 김치와 돼지고기, 두부를 넣고 끓이는 한국의 대표적인 국물 요리입니다. "
        "육수는 멸치와 다시마로 내며 마늘, 대파, 청양고추를 함께 넣어 얼큰한 맛을 냅니다. "
        "밥과 함께 먹는 것이 일반적입니다.",
        "ko", "food",
    ),
    f"{TEST_DOC_PREFIX}ko-legal": (
        "pipa-article-17.md",
        "개인정보보호법 제17조는 개인정보처리자가 개인정보를 제3자에게 제공할 때에는 "
        "정보주체의 동의를 받아야 함을 규정합니다. 다만 법률에 특별한 규정이 있는 경우 등 "
        "예외적인 사유에 해당하는 때에는 동의 없이 제공할 수 있습니다.",
        "ko", "legal",
    ),
    f"{TEST_DOC_PREFIX}en-python": (
        "python-asyncio.md",
        "Python 3.11 introduces asyncio.TaskGroup, a context manager for structured concurrency. "
        "TaskGroup guarantees that all spawned tasks either complete successfully or are cancelled "
        "together when the first exception is raised. It replaces older asyncio.gather patterns.",
        "en", "programming",
    ),
    f"{TEST_DOC_PREFIX}en-ml": (
        "transformers-attention.md",
        "The Transformer architecture uses self-attention to model dependencies between tokens "
        "in a sequence without relying on recurrence. Each attention head learns different "
        "relational patterns, and multi-head attention aggregates their outputs.",
        "en", "ml",
    ),
    f"{TEST_DOC_PREFIX}en-cloud": (
        "aws-vpc.md",
        "An AWS VPC is a logically isolated virtual network. Security Groups act as stateful "
        "firewalls attached to instances, while Network ACLs are stateless and evaluated per "
        "subnet. Traffic must pass both for inbound access.",
        "en", "cloud",
    ),
}


QUERIES: list[tuple[str, str | None, str]] = [
    ("탄군소프트가 어디에 있는 회사야?", f"{TEST_DOC_PREFIX}ko-company", "ko-paraphrase"),
    ("김치찌개 어떻게 끓여?", f"{TEST_DOC_PREFIX}ko-recipe", "ko-paraphrase"),
    ("개인정보를 제3자에게 넘길 때 필요한 것", f"{TEST_DOC_PREFIX}ko-legal", "ko-paraphrase"),
    ("What does asyncio.TaskGroup do?", f"{TEST_DOC_PREFIX}en-python", "en-paraphrase"),
    ("How does the attention mechanism in Transformers work?", f"{TEST_DOC_PREFIX}en-ml", "en-paraphrase"),
    ("difference between Security Group and NACL", f"{TEST_DOC_PREFIX}en-cloud", "en-paraphrase"),
    ("Where is TangunSoft headquartered?", f"{TEST_DOC_PREFIX}ko-company", "cross-en->ko"),
    ("How to cook Korean kimchi stew?", f"{TEST_DOC_PREFIX}ko-recipe", "cross-en->ko"),
    ("파이썬의 구조화된 동시성 패턴", f"{TEST_DOC_PREFIX}en-python", "cross-ko->en"),
    ("어텐션 메커니즘이 뭐야?", f"{TEST_DOC_PREFIX}en-ml", "cross-ko->en"),
    ("내일 서울 날씨 어때?", None, "adversarial"),
    ("best pizza toppings", None, "adversarial"),
]


DIRECT_PAIRS: list[tuple[str, str, tuple[float, float], str]] = [
    ("탄군소프트는 서울에 본사를 둔 소프트웨어 회사입니다.",
     "탄군소프트는 서울에 위치한 SW 개발사입니다.",
     (0.90, 1.00), "ko-paraphrase"),
    ("Python 3.11 introduces TaskGroup for structured concurrency.",
     "The TaskGroup class in Python 3.11 provides structured concurrency.",
     (0.85, 1.00), "en-paraphrase"),
    ("김치찌개는 김치와 돼지고기를 넣고 끓이는 국물 요리다.",
     "Kimchi stew is a Korean soup made with kimchi and pork.",
     (0.70, 1.00), "cross-lingual-same-meaning"),
    ("탄군소프트는 클라우드 컨설팅을 한다.",
     "TangunSoft does cloud infrastructure consulting.",
     (0.70, 1.00), "cross-lingual-same-meaning"),
    ("파이썬 asyncio 동시성 모델",
     "자바스크립트의 Promise 체이닝",
     (0.40, 0.80), "related-but-different"),
    ("탄군소프트 회사 소개",
     "라면 끓이는 방법",
     (0.0, 0.60), "unrelated"),
    ("Transformer self-attention",
     "kimchi stew ingredients",
     (0.0, 0.60), "unrelated"),
]


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _client() -> httpx.Client:
    return httpx.Client(timeout=120.0, verify=False, headers=_headers())


def part_a_direct() -> tuple[int, int]:
    print("=" * 70)
    print("PART A - Direct embedding similarity  (via /rag/debug/similarity)")
    print("=" * 70)
    print()

    passed = 0
    with _client() as c:
        print(f"  {'tag':<32} {'sim':>7}  {'expected':>13}  verdict")
        print(f"  {'-' * 32} {'-' * 7}  {'-' * 13}  -------")
        for a, b, (lo, hi), tag in DIRECT_PAIRS:
            r = c.post(f"{BASE}/rag/debug/similarity", json={"texts": [a, b]})
            r.raise_for_status()
            sim = float(r.json()["similarity"][0][1])
            ok = lo <= sim <= hi
            passed += int(ok)
            print(f"  {tag:<32} {sim:>7.3f}  [{lo:.2f}, {hi:.2f}]  {'PASS' if ok else 'FAIL'}")
    print()
    print(f"  Part A: {passed}/{len(DIRECT_PAIRS)} pair thresholds satisfied")
    print()
    return passed, len(DIRECT_PAIRS)


@dataclass
class QueryResult:
    query: str
    tag: str
    expected: str | None
    top1_doc: str
    top1_score: float
    top3: list[tuple[str, float]]


def part_b_retrieval() -> tuple[int, int]:
    print("=" * 70)
    print("PART B - End-to-end retrieval  (ingest -> search)")
    print("=" * 70)
    print(f"  BASE = {BASE}")
    print()

    with _client() as c:
        r = c.get(f"{BASE}/rag/health")
        r.raise_for_status()
        print(f"  health: {r.json()}")
        print()

        docs = c.get(f"{BASE}/rag/docs").json()
        leftovers = [d["doc_id"] for d in docs if d["doc_id"].startswith(TEST_DOC_PREFIX)]
        for did in leftovers:
            c.delete(f"{BASE}/rag/docs/{did}")
        if leftovers:
            print(f"  cleaned {len(leftovers)} leftover eval docs")

        print(f"  ingesting {len(CORPUS)} labeled docs...")
        t0 = time.perf_counter()
        for doc_id, (source, text, _lang, _topic) in CORPUS.items():
            r = c.post(
                f"{BASE}/rag/ingest/text",
                json={"doc_id": doc_id, "source": source, "text": text},
            )
            r.raise_for_status()
        print(f"  ingested in {time.perf_counter() - t0:.1f}s")
        print()

        results: list[QueryResult] = []
        print(f"  {'tag':<20} {'top1':>7}  {'gap':>6}  hit  query")
        print(f"  {'-' * 20} {'-' * 7}  {'-' * 6}  ---  ---------")
        for q, expected, tag in QUERIES:
            r = c.post(f"{BASE}/rag/search", json={"query": q, "k": 3})
            r.raise_for_status()
            hits = r.json()["hits"]
            if not hits:
                print(f"  {tag:<20} {'-':>7}  {'-':>6}  NO   {q[:42]}")
                continue
            top1_doc = hits[0]["doc_id"]
            top1_score = hits[0]["score"]
            top2_score = hits[1]["score"] if len(hits) > 1 else 0.0
            gap = top1_score - top2_score
            top3 = [(h["doc_id"], h["score"]) for h in hits[:3]]
            if expected is None:
                hit1 = "OK" if top1_score < 0.75 else "!!"
            elif top1_doc == expected:
                hit1 = "@1"
            elif expected in [d for d, _ in top3]:
                hit1 = "@3"
            else:
                hit1 = "MI"
            q_short = q if len(q) < 42 else q[:39] + "..."
            print(f"  {tag:<20} {top1_score:>7.3f}  {gap:>6.3f}  {hit1:<3}  {q_short}")
            results.append(QueryResult(q, tag, expected, top1_doc, top1_score, top3))

        print()
        _report_part_b(results)
        print()
        print("  cleaning up test corpus...")
        for doc_id in CORPUS:
            c.delete(f"{BASE}/rag/docs/{doc_id}")
        print(f"  removed {len(CORPUS)} eval docs")
        return _score_part_b(results)


def _report_part_b(results: list[QueryResult]) -> None:
    def hit_rate(rs: list[QueryResult], k: int) -> tuple[int, int]:
        hits = 0
        total = 0
        for r in rs:
            if r.expected is None:
                continue
            total += 1
            docs = [d for d, _ in r.top3[:k]]
            if r.expected in docs:
                hits += 1
        return hits, total

    print("  ---- aggregate ----")
    labeled = [r for r in results if r.expected is not None]
    adv = [r for r in results if r.expected is None]

    h1, t1 = hit_rate(labeled, 1)
    h3, t3 = hit_rate(labeled, 3)
    print(f"    Hit@1 (all labeled):        {h1}/{t1}  ({100*h1/max(t1,1):.0f}%)")
    print(f"    Hit@3 (all labeled):        {h3}/{t3}  ({100*h3/max(t3,1):.0f}%)")

    same_lang = [r for r in labeled if not r.tag.startswith("cross")]
    cross = [r for r in labeled if r.tag.startswith("cross")]
    hs1, ts1 = hit_rate(same_lang, 1)
    hc1, tc1 = hit_rate(cross, 1)
    print(f"    Hit@1 same-language:        {hs1}/{ts1}  ({100*hs1/max(ts1,1):.0f}%)")
    print(f"    Hit@1 cross-lingual:        {hc1}/{tc1}  ({100*hc1/max(tc1,1):.0f}%)")

    correct_scores = [r.top1_score for r in labeled if r.expected == r.top1_doc]
    if correct_scores:
        print(f"    avg top-1 score (correct):     {sum(correct_scores)/len(correct_scores):.3f}")
    if adv:
        print(f"    avg top-1 score (adversarial): {sum(r.top1_score for r in adv)/len(adv):.3f}  (should be lower)")


def _score_part_b(results: list[QueryResult]) -> tuple[int, int]:
    passed = 0
    for r in results:
        if r.expected is None:
            if r.top1_score < 0.75:
                passed += 1
        else:
            if r.expected in [d for d, _ in r.top3]:
                passed += 1
    return passed, len(results)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "all"], default="all")
    args = ap.parse_args()

    a_p = a_t = b_p = b_t = 0
    if args.part in ("a", "all"):
        a_p, a_t = part_a_direct()
    if args.part in ("b", "all"):
        b_p, b_t = part_b_retrieval()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if a_t:
        print(f"  Part A (direct embedding pairs):    {a_p}/{a_t}")
    if b_t:
        print(f"  Part B (retrieval queries):         {b_p}/{b_t}")
    total_p, total_t = a_p + b_p, a_t + b_t
    print(f"  TOTAL:                              {total_p}/{total_t}")
    print()
    print("  Score interpretation guide:")
    print("    >= 0.85    near-paraphrase or same meaning")
    print("    0.70-0.85  strong semantic match")
    print("    0.55-0.70  weak semantic match")
    print("    < 0.55     probably unrelated")
    print()
    return 0 if total_p == total_t else 1


if __name__ == "__main__":
    sys.exit(main())
