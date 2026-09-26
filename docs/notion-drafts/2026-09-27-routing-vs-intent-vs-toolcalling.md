# 라우팅 로직 · Intent 감지 · LLM Tool Calling — 세 개는 다른 층위다

> **상태**: seed
> **챕터**: 5. 라우팅 · Intent · Tool Calling
> **글감 발견 세션**: 2026-09-27, #37 아키텍처 상담 중

## 오해

"라우팅 로직 = intent 감지 = LLM tool calling 아니야?"

세 용어가 같은 것을 다르게 부르는 줄 알았다. Intent 감지가 tool calling이면, 라우팅 로직도 tool calling인가?

## 이해

**세 개는 포함 관계.** 라우팅 로직이 가장 크고, intent 감지가 그 안의 한 단계, tool calling은 그 단계의 **구현 방식** 중 하나.

```
┌─ 라우팅 로직 (전체 파이프라인) ────────────────────┐
│                                                     │
│  1. Protocol guard (코드 분기, LLM 없음)            │
│     model==mellum-4b? tools 있음? tool 히스토리?    │
│     → 조건 맞으면 plain pass-through                 │
│                                                     │
│  2. Intent 감지 (분류)  ← 여기서 방식 선택          │
│      B0: 키워드 (LLM 없음)                          │
│      B1: 별도 소형 분류 LLM                          │
│      B2: 메인 LLM tool calling ← 이걸 씀            │
│                                                     │
│  3. Dispatch (코드 분기)                            │
│     plain    → LiteLLM 직행                         │
│     rag      → /rag service                         │
│     ontology → /ontology service                    │
└─────────────────────────────────────────────────────┘
```

## 핵심: "분류용 tool calling"과 "답변용 tool calling"은 다르다

같은 tool calling 기법이지만 목적이 다르다.

### 답변용 (기존 온톨로지)
```python
tools = [list_customers, list_contracts, list_tickets, ...]
# LLM이 SQL 조회 tool을 골라 실행 → 결과로 진짜 답변 생성
```

### 분류용 (신규 intent classifier)
```python
tools = [route_plain, route_rag, route_ontology]
tool_choice = "required"     # 반드시 하나 호출
temperature = 0              # 결정적
max_tokens = 20              # 짧게
# LLM 응답: tool_calls: [{name: "route_ontology"}]
# 이건 "답변"이 아니라 "결정". 실제 답변은 dispatch된 곳에서 생성.
```

**왜 tool calling으로 분류하나:**
- 일반 프롬프트("plain/rag/ontology 중 하나 답해")는 LLM이 설명 붙이고 형식 어긋남.
- `tool_choice="required"` + JSON 구조화 → 파싱 무결.
- `temperature=0` + 짧은 `max_tokens` → 결정적, 빠름.

## 곁가지

- 이 관점은 tool calling을 **답변 생성용 API**로만 배운 사람에게 생소하다.
- OpenAI 문서는 tool calling을 "function calling"이라 부르는데, 사실은 "structured output enforcement"에 가깝다. 답변용/분류용/검증용 모두 같은 메커니즘 재활용.

## 확장할 때 다룰 것

- B0(키워드)가 실패한 실제 사례 (예: F3 wiki-fallback, "MongoDB는 어때?" 벤더 질문)
- B1(별도 소형 분류기)이 폐쇄망에서 비추인 이유 (GPU 추가 부담)
- B2에서 "LLM이 plain을 안 고르고 rag를 과호출하는 편향" 억제법
