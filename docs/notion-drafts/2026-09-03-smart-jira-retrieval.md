# Jira 이슈 스마트 검색 - Status 필터 + Recency Boost + Metadata 자동 추출

> Jira 이슈 78개 중 83% 가 완료(Done) 상태인 상황에서, 사용자가 "지금 진행 중인 작업이 뭐야?" 물어봐도 완료 이슈만 검색되어 "자료에 없습니다" 라고 잘못 답변하는 문제. Metadata 자동 추출 + 의도 기반 필터링 + 최신순 부스트 로 해결.

## 문제 상황

사내 RAG 시스템에서 Jira 이슈 검색이 "센스 없이" 동작한다.

**Test 사례 4가지**:

| Query | 답변 요지 | Top 5 status |
|---|---|---|
| "요즘 무슨 일 하고 있어?" | "USB 구매 완료... 삼성 차 완료..." | 5/5 완료 |
| "할 일 리스트 보여줘" | 신입 온보딩 문서로 회피 | 4/5 완료 |
| **"지금 진행 중인 작업이 뭐야?"** | **"자료에 없습니다" (오답)** | 5/5 완료 ⚠️ |
| "삼성 관련 이슈 있어?" | 5개 모두 "완료되었습니다" | 5/5 완료 |

**가장 심각한 케이스**: "지금 진행 중인 작업" 물어봤을 때 DB 에 진행 중 이슈 13개가 실제로 존재하는데도 검색이 완료 이슈만 뽑아와서 LLM 이 "자료에 없다" 로 답변. 사용자가 원하는 정보를 완전히 놓침.

---

## 데이터 실태

78개 Jira 이슈의 status 분포:

| Status | 개수 | 비중 |
|---|---|---|
| 완료 | ~65 | **83%** |
| 진행 중 | ~13 | 8% |
| 해야 할 일 | ~9 | 5% |
| 보류 | ~6 | 4% |

**압도적으로 완료 위주**. 스타트업 이슈 트래커 특성상 자연스러움 (해결된 게 대부분). 하지만 semantic-only 검색이 이 분포를 그대로 반영해서 항상 완료만 나옴.

---

## 근본 원인

### 문제 1: Jira export 는 텍스트만 씀

`scripts/exporters/jira_export.py` 는 status/created/priority 를 **마크다운 본문**에 인라인 텍스트로 넣는다:

```markdown
# [MAN-1] 삼성 DS 데모 준비
- **Status:** 완료
- **Priority:** High
- **Created:** 2026-04-13
- **Updated:** 2026-05-08
```

파싱 가능한 필드가 다 있지만 **구조화되지 않은 텍스트** 상태.

### 문제 2: rag-ingest 는 파싱 없음

`scripts/rag_ingest/core.py` 는 마크다운을 통째로 `/rag/ingest/text` 로 전송. 제목만 추출하고 나머지 필드는 무시.

### 문제 3: Chroma metadata 에 필터 필드 없음

`server/app/store.py:76-85` 저장 시 metadata:
```python
metadatas = [{
    "doc_id": doc_id, "chunk_index": i, "source": source, "added_at": ts,
    **base_metadata,  # 클라이언트가 넘긴 값 - 현재는 비어있음
}]
```

`base_metadata` 가 비어 있어서 Chroma 에 저장되는 필드는 시스템 메타 (hash, pool_path, doc_id) 뿐. **status/date 로 필터·정렬이 불가능한 상태**.

### 문제 4: retrieve 는 semantic-only

`server/app/rag.py:45-55`:
```python
def retrieve(query, k=None, where=None):
    vec = embed.embed_query(query)
    hits = get_store().query(vec, k=k or settings.top_k, where=where)
    ...
```

- Cosine similarity 로만 정렬
- Recency (created 최신) 무관
- 사용자 의도 (할 일 vs 완료) 감지 없음

83% 완료 DB 에서 top-5 뽑으면 대부분 완료 나오는 게 필연적 결과.

---

## 3단계 해결 설계

### D-1: Metadata 자동 추출 (server 측 파서)

**새 모듈**: `server/app/jira_meta.py`

```python
"""Extract structured metadata from Jira-exported markdown for filtered retrieval."""
import re
from datetime import datetime

_KEY_RE = re.compile(r"^#\s*\[(?P<key>[A-Z]+-\d+)\]", re.MULTILINE)
_FIELD_RE = re.compile(
    r"^-\s*\*\*(?P<field>[^:*]+):\*\*\s*(?P<value>.+?)\s*$",
    re.MULTILINE,
)

# Status categorization for cross-language filtering
_STATUS_COMPLETED = {"완료", "Done", "Closed", "Resolved", "닫힘"}
_STATUS_IN_PROGRESS = {"진행 중", "진행중", "In Progress"}
_STATUS_TODO = {"해야 할 일", "해야할일", "To Do", "Open", "New"}
_STATUS_ON_HOLD = {"보류", "On Hold", "Blocked"}


def _normalize_status(s: str) -> str:
    s = s.strip()
    if s in _STATUS_COMPLETED: return "completed"
    if s in _STATUS_IN_PROGRESS: return "in_progress"
    if s in _STATUS_TODO: return "todo"
    if s in _STATUS_ON_HOLD: return "on_hold"
    return "other"


def _parse_date(s: str) -> int | None:
    try:
        return int(datetime.strptime(s.strip(), "%Y-%m-%d").timestamp())
    except (ValueError, TypeError):
        return None


def parse_jira_metadata(text: str) -> dict | None:
    """Detect Jira-format markdown and extract structured metadata.
    
    Returns None if text doesn't look like a Jira issue.
    Otherwise returns dict with jira_key, jira_status, jira_created_ts, etc.
    """
    if not _KEY_RE.search(text):
        return None
    
    fields = {}
    for m in _FIELD_RE.finditer(text):
        fields[m.group("field").strip().lower()] = m.group("value").strip()
    
    key_m = _KEY_RE.search(text)
    meta = {"jira_key": key_m.group("key")}
    
    if raw_status := fields.get("status"):
        meta["jira_status_raw"] = raw_status
        meta["jira_status"] = _normalize_status(raw_status)
    if priority := fields.get("priority"):
        meta["jira_priority"] = priority
    if assignee := fields.get("assignee"):
        meta["jira_assignee"] = assignee
    if issuetype := fields.get("type"):
        meta["jira_type"] = issuetype
    if created := fields.get("created"):
        if ts := _parse_date(created):
            meta["jira_created_ts"] = ts
            meta["jira_created"] = created
    if updated := fields.get("updated"):
        if ts := _parse_date(updated):
            meta["jira_updated_ts"] = ts
            meta["jira_updated"] = updated
    
    return meta
```

**통합 지점**: `main.py::_index()` 시작부

```python
from .jira_meta import parse_jira_metadata

def _index(text: str, source: str, doc_id: str | None, metadata: dict) -> IngestResponse:
    # Phase D-1: auto-extract Jira metadata if this is a Jira export
    if jira_meta := parse_jira_metadata(text):
        metadata = {**metadata, **jira_meta}  # merge, client explicit wins
    ...
```

**결과**: 앞으로 ingest 되는 모든 Jira 문서는 `jira_status`, `jira_created_ts` 등이 Chroma metadata 로 검색·필터 가능.

---

### D-2: Smart Retrieval (server 측 검색 로직)

**의도 감지**: 사용자 쿼리에서 "할 일" 계열 키워드 추출

```python
_ACTIVE_WORK_KW = [
    "할 일", "할일", "진행", "무슨 일", "무슨일",
    "요즘", "최근", "이번 주", "이번주", "이번 달", "이번달",
    "지금", "현재", "todo", "TODO", "미완료",
]
_EXPLICIT_COMPLETED_KW = [
    "완료된", "끝난", "마친", "완성된", "완료 이슈", "완료된 이슈",
]


def _detect_active_work_intent(query: str) -> bool:
    """User asking about ongoing work vs completed work?"""
    has_active = any(kw in query for kw in _ACTIVE_WORK_KW)
    has_explicit_completed = any(kw in query for kw in _EXPLICIT_COMPLETED_KW)
    # 명시적 "완료된" 이 있으면 사용자가 완료를 원함 → 필터 off
    return has_active and not has_explicit_completed
```

**Recency boost**: 최신 이슈 우선

```python
import time


def _apply_recency_boost(score: float, meta: dict) -> float:
    """Boost score by recency of jira_created_ts (6 months window, max 15%)."""
    created_ts = meta.get("jira_created_ts")
    if not created_ts:
        return score
    days_ago = (time.time() - created_ts) / 86400
    boost = max(0.0, 0.15 * (1 - days_ago / 180))
    return score + boost
```

**Retrieve 재작성**:

```python
def retrieve(query, k=None, where=None):
    vec = embed.embed_query(query)
    k_actual = k or settings.top_k
    
    # Phase D-2: 의도 감지 시 오버페치 + Python 필터
    intent = _detect_active_work_intent(query) and not where
    k_fetch = k_actual * 3 if intent else k_actual
    
    hits = get_store().query(vec, k=k_fetch, where=where)
    
    if settings.min_score > 0:
        hits = [h for h in hits if h["score"] >= settings.min_score]
    
    # Filter completed (Python side to avoid Chroma "no field = excluded" trap)
    if intent:
        hits = [h for h in hits
                if h.get("metadata", {}).get("jira_status") != "completed"]
    
    # Recency boost + resort
    for h in hits:
        h["score"] = _apply_recency_boost(h["score"], h.get("metadata", {}))
    hits.sort(key=lambda h: h["score"], reverse=True)
    
    return hits[:k_actual]
```

**왜 Python 필터인가**: Chroma 의 `$ne` 는 필드가 없는 문서를 제외한다. 정책 문서, 신입 온보딩 등 Jira 아닌 문서에는 `jira_status` 필드 자체가 없기 때문에 Chroma `where` 로 필터하면 그것들까지 다 배제됨. Python 에서 필터하면 필드 없는 문서는 자연히 통과 (`get(...) != "completed"` → True).

**왜 3배 오버페치인가**: 완료가 83% 니 필터 후 부족할 위험. 3배 뽑고 필터 → 리랭킹 → top-k.

---

### D-3: 기존 78개 Backfill

새 파서는 앞으로 ingest 만 커버. 기존 78개 문서는 metadata 없어서 필터/부스트 대상 안 됨.

**해결**: 78개 재 ingest.

가장 깨끗한 방법 - sync-state.json 에서 jira-* 엔트리만 제거하면 다음 sync 때 NEW 로 처리됨:

```bash
# On EC2
sudo systemctl stop rag-sync.timer

python3 -c "
import json, pathlib
p = pathlib.Path('/upload/rag/data/sync-state.json')
state = json.loads(p.read_text())
before = len(state.get('ingested', {}))
state['ingested'] = {k: v for k, v in state['ingested'].items()
                    if not k.startswith('jira-')}
after = len(state['ingested'])
print(f'Removed {before - after} jira entries from state')
p.write_text(json.dumps(state, ensure_ascii=False, indent=2))
"

sudo -u rocky /upload/rag/venv/bin/rag-ingest sync --limit 100
```

- Store.add() 는 `delete(doc_id, prune_registry=False)` 로 기존 chunks 삭제 후 재 삽입
- 새 metadata (jira_status 등) 가 붙은 상태로 재저장
- 78개 * ~500ms = 약 40초 소요 예상

---

## 예상 효과 (Before/After)

| Test | Query | Before | After (예상) |
|---|---|---|---|
| 1 | "지금 진행 중 작업?" | "자료에 없습니다" (오답, 완료만 매칭) | 진행 중 3개 리스트 |
| 2 | "요즘 무슨 일?" | 완료 5개 "완료했습니다" | 진행 중 + 최근 이슈 mix, recency 우선 |
| 3 | "할 일 리스트" | 온보딩 문서로 회피 | 해야 할 일 status 이슈들 |
| 4 | "삼성 관련 이슈 있어?" | 완료 5개 (의도 감지 안 됨) | 그대로 유지 (사용자가 status 안 물음) |
| 5 | "완료된 이슈 뭐 있어?" | 완료 5개 | 완료 5개 유지 (명시적 요청) |
| 6 | 임의 정책 문서 쿼리 | 정상 | 정상 (jira_status 없어서 필터 안 걸림) |

---

## Phase A/B/C 와의 관계

| Phase | 대상 | 방어층 |
|---|---|---|
| A | 한자 leak | 응답 사후 재시도 |
| B | 학습해 트리거 | Ingest 사전 감지 |
| C | Bridging hallucination | Retrieval 사전 grounding 통합 |
| **D** | **Jira 이슈 스마트 검색** | **Ingest 시 metadata 추출 + Retrieval 시 의도 기반 필터** |

D 는 다른 Phase 와 완전히 독립. 다른 phase 방어층에 영향 없이 metadata 만 추가.

---

## 성공 기준

배포 + 78개 backfill 후 다음 6개 케이스 통과:

| # | 쿼리 | 기대 결과 |
|---|---|---|
| 1 | "지금 진행 중인 작업이 뭐야?" | 진행 중 이슈만 반환 (완료 0개) |
| 2 | "요즘 무슨 일 하고 있어?" | 완료 없이 진행 중 + 해야 할 일 |
| 3 | "할 일 리스트 보여줘" | 해야 할 일 status 우선 |
| 4 | "삼성 관련 이슈 있어?" | 필터 없이 semantic 검색 그대로 |
| 5 | "완료된 이슈 뭐 있어?" | 완료 이슈 반환 (명시적 요청) |
| 6 | Chroma metadata 에 jira_status/jira_created_ts 존재 | 78개 전부 |

---

## 정리

- Jira export 는 데이터가 풍부하지만 마크다운 텍스트에 갇혀 있어서 필터·정렬 불가
- Metadata 추출 로직 하나 추가하면 Chroma 의 강력한 `where` 필터 활용 가능해짐
- 사용자 쿼리의 자연어 의도를 감지해서 자동 필터 적용 (수동 파라미터 필요 없음)
- Recency boost 로 오래된 이슈보다 최신 우선
- 기존 78개는 state 리셋 후 재 sync 로 자동 마이그레이션

**핵심 원칙**: RAG 는 semantic 유사도만으로 부족하다. 문서 구조가 있으면 그걸 활용해서 필터·정렬해야 진짜 "스마트"한 검색이 됨. Semantic + Structured 조합이 정답.
