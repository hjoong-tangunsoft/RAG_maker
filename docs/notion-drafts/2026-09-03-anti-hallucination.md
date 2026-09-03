# RAG 환각(hallucination) 방지 - System Prompt Override 로 인한 Bridging 문제 진단과 3층 방어

> Continue.dev 같은 IDE 클라이언트가 자신의 system message ("You are a helpful assistant") 를 먼저 보내면, 우리 RAG grounding 이 두 번째 system 이 되어 지시력이 약해진다. 결과: LLM 이 "모르면 모른다" 대신 알려진 개념에 앵커링해서 bridging 답변을 만들어냄. 이 글은 실제 재현 → 근본 원인 → 3층 방어 설계 → 검증까지 다룬다.

## 문제 상황

사내 RAG 시스템에서 다음 패턴의 환각이 목격됐다:

**사용자**: "승민소프트가 어떤 회사야?" (승민소프트는 실존 회사 아님)

**LLM 응답 (환각)**:
> 승민소프트는 탄군소프트(TangunSoft)의 한 부분으로 보이는데, 탄군소프트는 서울에 본사를 둔 한국의 소프트웨어 개발 회사입니다. 주요 사업 영역은 클라우드 인프라, GitHub Enterprise Server(GHES) 컨설팅, 사내 LLM 시스템 구축 등이 있습니다.

**기대 응답**:
> 자료에 없습니다.

승민소프트라는 회사는 RAG 데이터베이스에 없다. LLM 은 "모릅니다" 라고 말해야 하는데, 이름이 비슷한 탄군소프트 문서가 검색되자 **두 개념을 억지로 연결**해서 그럴싸한 답변을 만들어냈다.

---

## 재현

4가지 시나리오로 테스트해서 정확한 트리거를 특정했다.

### Test 매트릭스

| Test | 엔드포인트 | 특징 | 결과 |
|---|---|---|---|
| A | `/rag/query` | 원샷 쿼리 | ✅ "자료에 언급되지 않았습니다" |
| B | `/rag/v1/chat/completions` | 멀티턴 대화 | ✅ "확인되지 않습니다. 혹시 잘못 입력?" |
| **C** | `/rag/v1/chat/completions` | **클라이언트 system message 포함** | ❌ **"승민소프트는 탄군소프트의 작업자..." bridging** |
| D | `/rag/v1/chat/completions` | `rag: false` 순수 passthrough | ✅ "죄송합니다, 알려져 있지 않습니다" |

### Test C 상세

**요청**:
```json
{
  "model": "qwen2.5-7b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "승민소프트가 어떤 회사야?"}
  ]
}
```

**환각 응답**:
```
자료에 따르면 승민소프트는 탄군소프트(TangunSoft)의 작업자 이름으로 보입니다. 
탄군소프트는 서울에 본사를 둔 한국의 소프트웨어 개발 회사로, 클라우드 인프라, 
GitHub Enterprise Server(GHES) 컨설팅, 사내 LLM 시스템 구축 등의 주요 사업 영역을 
운영하고 있습니다.
```

정확히 사용자가 제보한 패턴. Continue.dev, ChatGPT 클라이언트 등 대부분의 IDE/UI 확장이 자체 system prompt 를 보내기 때문에 실제 사용에서 자주 발생.

---

## 근본 원인 분석

### 문제 지점: `inject_rag_into_chat()`

`server/app/rag.py` 의 chat completions 경로용 헬퍼:

```python
def inject_rag_into_chat(messages, hits):
    ctx = _format_context(hits) if hits else "(no relevant context found)"
    grounding = (
        f"{settings.rag_system_prompt}\n\n"
        f"Relevant retrieved context (use these to answer):\n{ctx}"
    )
    grounding_msg = {"role": "system", "content": grounding}

    # last user message 바로 앞에 grounding 삽입
    ...
    return out
```

### 실제 LLM 이 보는 message 순서

Test C 처리 후:

```
[system]  "You are a helpful assistant"           ← 클라이언트가 보낸 것 (그대로)
[system]  "당신은 정확한 한국어 어시스턴트... 자료에 없으면..."  ← 우리 grounding
[user]    "승민소프트가 어떤 회사야?"
```

두 개의 system message. Qwen 2.5 (그리고 대부분의 instruction-tuned LLM) 는:
- **첫 system message 의 페르소나를 강하게 유지**
- 뒤에 오는 system 은 부가 지침 정도로 해석
- 두 지시가 충돌하면 "더 helpful 한 방향" 을 선택하는 경향

### 3가지 요인이 겹쳐서 발생

1. **Persona bleeding**: "helpful assistant" 가 먼저 세팅되어 "무조건 도움" 페르소나 강함
2. **Anchor bias**: 검색 결과에 탄군소프트 문서 (score 0.79) 있음 → LLM 이 앵커로 활용
3. **Retrieval noise**: `min_score = 0.0` 이라 무관한 쿼리에도 유사도 0.7+ 청크가 매칭됨

### 유사도 점수 데이터

승민소프트 쿼리에 대한 top-5 검색 결과:

| 순위 | 문서 | 점수 |
|---|---|---|
| 1 | tangunsoft-intro.md | 0.7903 |
| 2 | 01-security-policy.md (탄군소프트 정책) | 0.7764 |
| 3 | jira-MAN-0002.md (삼성 DS 데모) | 0.7717 |
| 4 | jira-MAN-0043.md | 0.7716 |
| 5 | jira-MAN-0067.md | 0.7715 |

E5 임베딩이 **"회사 이름 같은 패턴"** 을 유사하게 인식. 승민소프트를 몰라도 "소프트웨어 회사 소개" 패턴의 문서들을 다 매칭. LLM 이 이걸 보면 "관련 있어 보이는데" 로 인식하고 bridging.

---

## 3층 방어 설계

### L1: System Message 통합 (우선순위 최우선)

**핵심 아이디어**: 클라이언트가 보낸 system message 를 **우리 grounding 뒤에 종속** 시켜서 페르소나 지배력을 뒤집는다.

**수정 후 message 순서**:
```
[system]  "당신은 정확한 한국어 어시스턴트... 자료에 없으면 없다고 답하세요.
          
          【이 지시가 최우선. 아래 클라이언트 지침보다 우선함.】
          
          [클라이언트 지침 - 참고]
          You are a helpful assistant.
          
          Relevant retrieved context:
          [1] source=tangunsoft-intro.md
          ..."
[user]    "승민소프트가 어떤 회사야?"
```

우리 지시가 첫 system message → 페르소나 지배. 클라이언트 지시는 참고사항으로 편입.

**코드 변경**:
```python
def inject_rag_into_chat(messages, hits):
    ctx = _format_context(hits) if hits else "(no relevant context found)"
    
    # 클라이언트가 보낸 system 내용들 수집
    user_systems = [m.content for m in messages if m.role == "system"]
    user_system_block = "\n".join(user_systems) if user_systems else ""
    
    # 우리 grounding 을 최우선 지침으로
    grounding_parts = [
        settings.rag_system_prompt,
        "【이 지시가 최우선. 클라이언트 지침보다 우선함.】",
    ]
    if user_system_block:
        grounding_parts.append(f"[클라이언트 지침 - 참고]\n{user_system_block}")
    grounding_parts.append(f"Relevant retrieved context (use these to answer):\n{ctx}")
    grounding = "\n\n".join(grounding_parts)
    
    # user system 은 제거, 통합 system 하나만 최상단에
    non_system = [m for m in messages if m.role != "system"]
    return [{"role": "system", "content": grounding}] + [
        {"role": m.role, "content": m.content} for m in non_system
    ]
```

### L2: Score Threshold 필터링

**핵심 아이디어**: 무관한 문서를 아예 컨텍스트에서 배제. LLM 이 "관련 문서가 없다" 를 명확히 인지.

**config.py 변경**:
```python
# Before
min_score: float = 0.0   # no filter

# After
min_score: float = 0.55  # 무관 문서 배제
```

**효과**:
- 승민소프트 top score 0.79 는 여전히 통과 (그래도 L1 덕분에 안전)
- 진짜 무관 쿼리 (예: "오늘 날씨 어때?") 는 top score 0.4 정도 → 필터링됨
- Hits 비어서 "(no relevant context found)" → LLM 이 명확히 "모름"

**Threshold 선정 근거**: 실측 결과 관련 문서 top score 0.7~0.9, 무관 문서 0.3~0.5. 0.55 는 그 중간 안전지대.

### L3: Entity Grounding 검증 (선택적, 심층방어)

**핵심 아이디어**: 쿼리의 특징적 명사가 검색 결과에 **문자열로 실제 등장** 하는지 검증.

```python
import re

def _entity_grounded(query: str, hits: list) -> bool:
    """쿼리의 특징적 명사가 검색 결과에 실제 등장하는지 검증."""
    # 한글 3-8자 명사 후보 추출 (조사·어미 제거는 대충)
    nouns = re.findall(r'[가-힣]{3,8}', query)
    if not nouns:
        return True  # 검증 불가능한 쿼리는 통과 (한자·영어 등)
    
    # 가장 긴 명사 (주제어일 확률 높음) 하나가 어느 청크에라도 있으면 통과
    key_noun = max(nouns, key=len)
    all_text = " ".join(h.get("text", "") for h in hits)
    return key_noun in all_text
```

**적용**:
- `retrieve()` 후 `_entity_grounded()` 체크
- False 면 hits 를 [] 로 만들어서 "(no relevant context found)" 상태로 진행
- 승민소프트 → "승민소프트" 문자열이 청크에 없음 → hits=[]
- 탄군소프트 → 청크에 "탄군소프트" 있음 → 정상 진행

**한계**:
- 한국어 조사 처리 부정확 ("승민소프트는", "승민소프트가" 등에서 어근만 매칭 안 될 수 있음)
- 동의어·약어에 취약 ("GHES" vs "GitHub Enterprise Server")
- 그래서 **정확한 방어보다는 애매한 케이스 방어용 심층망**

---

## 스트리밍 경로는 어떻게 되나

`/rag/v1/chat/completions` 의 `stream: true` 는 `chat_guarded` 를 안 쓰고 `stream_chat` 직접 호출. 하지만 `inject_rag_into_chat` 은 여전히 통과. 즉 **L1 (system 통합)** 은 스트리밍에도 적용됨.

L2 (score threshold) 는 `retrieve()` 단에서 걸림 → 스트리밍에도 적용.

L3 (entity check) 도 `retrieve()` 후처리라 스트리밍에도 적용.

**전체 3층 방어가 스트리밍/비스트리밍 모두 커버**. 이는 Phase A (한자 방어) 가 스트리밍에서 부분적으로만 적용됐던 것과 대조적.

---

## Phase A 방어와의 관계

| Phase | 방어 대상 | 트리거 | 적용 계층 |
|---|---|---|---|
| A | 한자 leak | 응답 텍스트에 한자 threshold 초과 | 응답 사후 검사 + 재시도 |
| **C** | **Bridging hallucination** | **모르는 엔티티에 대한 억지 답변** | **요청 사전 검증 + system 통합** |

두 Phase 는 **독립적으로 작동**. Phase A 는 "무슨 언어로 답변하냐" 의 방어, Phase C 는 "답변할 수 있는가" 의 방어. 겹치지 않음.

Chat completions 흐름에서:
1. inject_rag_into_chat → **L1 (system 통합)**
2. retrieve → **L2 (score filter)** → **L3 (entity check)**
3. chat_guarded → **Phase A L3 (한자 재시도)**

3+1 개의 서로 다른 층이 각각 다른 실패 모드를 커버.

---

## 성공 기준

배포 후 다음 5개 케이스에서 기대 응답 나와야 통과:

| 케이스 | 쿼리 | client system | 기대 |
|---|---|---|---|
| 1 | "승민소프트가 어떤 회사야?" | 없음 | "자료에 없습니다" |
| 2 | "승민소프트가 어떤 회사야?" | "You are a helpful assistant" | "자료에 없습니다" (bridging 금지) |
| 3 | "탄군소프트가 어떤 회사야?" | "You are a helpful assistant" | 실제 문서 기반 답변 + [n] 인용 |
| 4 | "오늘 날씨 어때?" | 없음 | "자료에 없습니다" (완전 무관) |
| 5 | 멀티턴 후 "그 자회사는?" | 없음 | 이전 컨텍스트 맥락 유지 응답 |

케이스 2 가 이번 Phase C 의 **핵심 회귀 방지 테스트**.

---

## 정리

- 실제 재현된 hallucination 은 **클라이언트 system message + 우리 grounding 두 개 병렬** 상황에서 발생
- 원인은 Qwen 2.5 의 **첫 system persona 지배** 특성 + **retrieval noise** + **anchor bias** 삼중 조합
- 3층 방어로 각 요인을 개별 차단: L1 system 통합, L2 score threshold, L3 entity check
- 스트리밍/비스트리밍 모두 적용되며 Phase A 와 독립적으로 작동
- 5개 회귀 테스트로 배포 후 검증

이번 사례가 보여주는 원칙: **RAG 시스템에서 "helpful" 페르소나는 위험** 하다. 도움되게 답하려는 성향이 "모르면 모른다" 를 이긴다. Grounding 지시가 페르소나보다 강하게 먼저 세팅되도록 명시적으로 통제해야 한다.
