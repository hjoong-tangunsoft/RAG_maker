# RAG 요청 흐름 6단계 - 사용자 질문 한 개가 답변으로 돌아오기까지

> 사용자가 `POST /rag/query` 로 질문 하나 던지면 내부에서 무슨 일이 벌어지는가. FastAPI 엔드포인트부터 vLLM Qwen 추론까지 6개 계층을 실제 함수 호출과 함께 순차 추적.

## 왜 이 글?

RAG 시스템 문서 대부분은 "검색 + 생성 결합" 정도로 개념만 설명하고 끝난다. 실제로 사용자 질문 하나가 시스템에 도착해서 답변으로 돌아오기까지 **몇 개의 프로세스, 몇 개의 함수, 몇 번의 HTTP 홉을 지나는지** 는 잘 다루지 않는다.

이 글은 우리 사내 RAG_maker 시스템에서 요청 한 개가 어떤 경로를 밟는지 코드 레벨로 추적한다. Phase A 에서 추가한 Korean Purity Guard (`chat_guarded`) 가 어디에 끼어드는지도 함께 표시한다.

**대상 독자**: 파이썬 서비스 개발자 초급. FastAPI 는 봐본 정도, RAG 는 처음 만지는 상태를 가정.

---

## 전체 흐름 한 장 요약

썸네일 이미지 참고. 6개 층을 관통한다:

| 단계 | 컴포넌트 | 파일/프로세스 | 포트 |
|---|---|---|---|
| 1 | FastAPI 엔드포인트 | `main.py` | `:8100` |
| 2 | RAG 오케스트레이터 | `rag.py` | in-process |
| 3 | Korean Purity Guard ⭐ | `llm.py chat_guarded` | wrapper |
| 4 | HTTP 클라이언트 | `llm.py chat` | httpx |
| 5 | LiteLLM 프록시 | 별도 프로세스 | `:4000` |
| 6 | vLLM + Qwen 추론 | 별도 프로세스 | `:8000` |

⭐ 표시가 Phase A 에서 신규 도입한 방어층.

---

## Step 1: FastAPI 엔드포인트 진입

### 무슨 일이 일어나는가

사용자가 다음 요청을 보낸다:

```bash
curl -X POST http://52.79.62.107:8100/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "회사 정책 요약해줘", "k": 5}'
```

FastAPI 가 이걸 받으면 아래 함수로 라우팅한다:

```python
# server/app/main.py
@app.post("/rag/query", response_model=QueryResponse)
async def rag_query(body: QueryRequest) -> QueryResponse:
    text, citations, used_model = await rag.answer(
        query=body.query,
        k=body.k,
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )
    return QueryResponse(
        answer=text,
        citations=citations,
        model=used_model,
    )
```

### 여기서 실제로 하는 일

1. **Pydantic 검증**: `QueryRequest(BaseModel)` 스키마가 `body` 필드를 자동 검사. `query` 가 빠지면 여기서 400 에러로 즉시 반환. 함수 안으로 못 들어옴.
2. **비동기 함수 진입**: `async def` 라서 다른 요청이 들어와도 이 함수가 vLLM 대기하는 동안 다른 요청 처리 가능.
3. **`rag.answer()` 위임**: 실제 RAG 로직은 `rag.py` 모듈에 있음. 엔드포인트는 "받아서 넘김" 만 담당.

### 왜 얇게 유지하는가

엔드포인트가 두꺼우면 다음이 안 됨:
- 같은 로직을 CLI 나 배치 잡에서 재사용
- rag.py 만 단위 테스트
- HTTP 관심사 (status code, 헤더) 와 도메인 관심사 (검색, LLM 호출) 분리

이 원칙 덕분에 `rag.answer()` 는 완전히 HTTP 를 모른다.

---

## Step 2: RAG 오케스트레이션

### `rag.answer()` 함수 뼈대

```python
# server/app/rag.py
async def answer(
    query: str,
    k: int = 5,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, list[Citation], str]:
    # (1) 벡터 검색
    hits = retrieve(query, k=k)

    # (2) 프롬프트 조립
    messages = build_rag_messages(query, hits)

    # (3) LLM 호출 (한자 방어 wrapper)
    resp = await llm.chat_guarded(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # (4) 응답 파싱 + 인용 조립
    text = resp["choices"][0]["message"]["content"]
    used_model = resp.get("model", "unknown")
    citations = [
        Citation(doc_id=h["doc_id"], score=h["score"], snippet=h["text"][:200])
        for h in hits
    ]
    return text, citations, used_model
```

### 4개 하위 단계 상세

#### 2-1. `retrieve(query, k=5)` - Chroma 벡터 검색

```python
def retrieve(query: str, k: int = 5) -> list[dict]:
    coll = get_collection()
    results = coll.query(query_texts=[query], n_results=k)
    hits = []
    for i, doc in enumerate(results["documents"][0]):
        hits.append({
            "doc_id": results["ids"][0][i],
            "score": results["distances"][0][i],
            "text": doc,
        })
    return hits
```

- Chroma 가 내부적으로 embedding 함수 (BGE-M3) 를 호출해서 query 를 벡터로 변환
- 저장된 문서 벡터들과 코사인 유사도 계산
- 상위 k 개 반환

**소요 시간**: 문서 수백~수천 개 규모면 보통 50-200ms.

#### 2-2. `build_rag_messages(query, hits)` - 프롬프트 조립

```python
def build_rag_messages(query: str, hits: list[dict]) -> list[dict]:
    context_block = "\n\n".join([
        f"[문서 {i+1}] {h['text']}"
        for i, h in enumerate(hits)
    ])
    return [
        {
            "role": "system",
            "content": (
                "당신은 정확한 한국어 어시스턴트입니다. "
                "제공된 문서를 기반으로만 답하세요. "
                "문서에 없는 내용은 '모릅니다'라고 답하세요. "
                "답변은 반드시 한국어로만 작성하세요."
            ),
        },
        {
            "role": "user",
            "content": f"{context_block}\n\n[질문]\n{query}",
        },
    ]
```

**핵심**: system 메시지에 "한국어로만" 을 명시. 이게 Phase A 의 L1 방어층 (프롬프트 강화). 그래도 뚫리는 경우가 있어서 Step 3 의 L3 방어가 필요.

#### 2-3. `llm.chat_guarded(...)` - 다음 단계로 진입 (Step 3)

#### 2-4. 응답 파싱

```python
text = resp["choices"][0]["message"]["content"]
```

이 딕셔너리 구조는 OpenAI API 표준. LiteLLM 이 Qwen 응답을 이 형식으로 정규화해서 반환하므로 우리 코드는 늘 같은 방식으로 파싱.

---

## Step 3: Korean Purity Guard (`chat_guarded`) ⭐

### Phase A 신규 wrapper 층

```python
# server/app/llm.py
async def chat_guarded(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
    max_retries: int = 1,
) -> dict[str, Any]:
    """chat() + retry once if response contains too many CJK ideographs."""
    resp = await chat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra=extra,
    )
    try:
        text = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return resp

    n = hanja_count(text)
    if n <= settings.hanja_threshold or max_retries <= 0:
        return resp

    log.warning(
        "hanja leak detected (%d chars, threshold=%d), regenerating",
        n, settings.hanja_threshold,
    )
    retry_messages = _reinforce_korean_only(messages)
    resp2 = await chat(
        retry_messages,
        model=model,
        temperature=0.05,
        max_tokens=max_tokens,
        extra=extra,
    )
    return resp2
```

### 이 층의 역할

- `chat()` 을 호출해서 답변을 받음
- **한자 개수** 를 셈 (`hanja_count`)
- threshold (기본 2) 초과면 강화된 프롬프트로 **한 번 재시도**
- 스트리밍은 재시도가 불가능해서 이 층을 안 씀 (`stream_chat` 은 별도 경로)

### 왜 여기 놓았는가

- `rag.py` 는 도메인 로직 (retrieve, prompt build) 담당. HTTP 관심사 없음
- `llm.py chat()` 은 순수 HTTP 클라이언트. 재시도 로직 없음
- **`chat_guarded()` 가 그 사이에 끼는 wrapper** 로서 두 관심사 분리 유지

### 실제 발동 확률

이번 Phase A 배포 후 실측: 5개 테스트 시나리오에서 hanja=0. Threshold 초과로 재시도 발동한 케이스 매우 드묾. 이유:
- Qwen 2.5-7B 는 한국어 학습 데이터 충분해서 대부분 한자 안 씀
- system 프롬프트에서 이미 강조
- temperature 0.1 이라 창의적 이탈 억제

즉 **99% 는 첫 호출에서 통과**. 이 층은 남은 1% 를 위한 안전망.

---

## Step 4: HTTP 클라이언트 `chat()`

### 실제 HTTP 요청 만드는 지점

```python
# server/app/llm.py
async def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model or settings.default_model,
        "messages": messages,
        "temperature": settings.rag_temperature if temperature is None else temperature,
        "max_tokens": settings.rag_max_tokens if max_tokens is None else max_tokens,
        "stream": False,
    }
    if extra:
        payload.update(extra)
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(_CHAT_URL, json=payload, headers=_headers())
        r.raise_for_status()
        return r.json()
```

### 조립되는 실제 payload

```json
{
  "model": "qwen2.5-7b",
  "messages": [
    {"role": "system", "content": "당신은 정확한 한국어 어시스턴트..."},
    {"role": "user", "content": "[문서 1] ...\n\n[질문]\n회사 정책 요약해줘"}
  ],
  "temperature": 0.1,
  "max_tokens": 1024,
  "stream": false
}
```

### 실제 HTTP 요청

```
POST http://127.0.0.1:4000/v1/chat/completions
Headers:
  Authorization: Bearer sk-1234
  Content-Type: application/json
Body: (위 JSON)
```

### 왜 httpx 인가

파이썬의 표준 HTTP 라이브러리인 `requests` 는 **동기 전용**. `await` 불가. FastAPI 처럼 async 환경에서는 `httpx.AsyncClient` 필수.

**동기 방식이면**: vLLM 이 답변 만드는 3-5초 동안 파이썬 워커 하나가 완전히 blocked. 동시에 100개 요청 오면 100개 워커 필요.
**비동기 방식이면**: 대기 중에 이벤트 루프가 다른 요청을 처리. 워커 하나가 100개 동시 요청 감당 가능.

### timeout=300 인 이유

Qwen 2.5-7B 로 긴 답변 만들 때 최대 30-60초. 여유 잡아서 5분. `raise_for_status()` 는 HTTP 4xx/5xx 면 예외 발생시켜서 상위 (`chat_guarded` → `rag.answer`) 에서 잡을 수 있게 함.

---

## Step 5: LiteLLM 프록시

### 왜 LiteLLM 이 중간에 있는가

우리 스택 구조:

```
FastAPI :8100 (RAG)
    ↓
LiteLLM :4000 (proxy)
    ↓
vLLM :8000 (Qwen inference)
```

LiteLLM 없이 FastAPI 가 직접 vLLM 호출해도 됨. 그런데 왜 중간에 프록시를 두는가:

| LiteLLM 이 하는 일 | 의미 |
|---|---|
| **모델 이름 매핑** | `qwen2.5-7b` (별칭) → `Qwen/Qwen2.5-7B-Instruct` (실제 모델 ID) |
| **인증 통제** | Bearer 토큰 검사. 잘못된 키면 401. |
| **OpenAI API 호환** | Continue.dev, ChatGPT 클라이언트 같은 앱이 그대로 붙음 |
| **여러 백엔드 로드 밸런싱** | 나중에 vLLM 인스턴스 여러 개 붙일 때 |
| **로깅/과금** | 모델별 토큰 사용량 집계 |

우리 케이스에서는 특히 **OpenAI API 호환** 이 크다. Continue.dev 확장이 그대로 붙어서 IDE 안에서 사내 문서 QA 가능해짐.

### LiteLLM 처리

1. Bearer 토큰 확인 (`sk-1234`)
2. `model` 필드 (`qwen2.5-7b`) 를 config 에서 lookup 해서 실제 vLLM 엔드포인트 찾음
3. Body 그대로 vLLM 으로 forward
4. vLLM 응답을 그대로 리턴

**소요 시간**: 순수 프록시 오버헤드는 10ms 미만. 대부분 vLLM 대기 시간.

---

## Step 6: vLLM + Qwen 추론

### GPU 에서 실제 토큰 생성

vLLM 은 별도 프로세스로 GPU 위에 Qwen 2.5-7B 모델을 로드해두고 있음. LiteLLM 이 forward 한 요청을 받아서:

1. **토크나이징**: 입력 텍스트 → 토큰 ID
2. **KV 캐시**: 이전 요청과 공통 prefix 는 재사용
3. **어텐션 계산**: GPU L4 에서 각 토큰의 attention weight 계산
4. **샘플링**: temperature 0.1 이므로 거의 greedy. 확률 최댓값 토큰 선택
5. **detokenize**: 토큰 ID → 텍스트
6. **stop 조건** 도달까지 반복 (`</s>`, max_tokens 등)

### 실측 성능

- Prompt 500 tokens, output 300 tokens 케이스: 약 3-5초
- Prompt 2000 tokens, output 800 tokens: 약 8-12초
- 병목은 대부분 **output 생성** (autoregressive 특성)

### 응답 형태

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1725000000,
  "model": "qwen2.5-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "탄군소프트의 정책은 다음과 같습니다: 첫째..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 543,
    "completion_tokens": 287,
    "total_tokens": 830
  }
}
```

이게 그대로 LiteLLM → httpx → `chat()` → `chat_guarded()` → `rag.answer()` → 엔드포인트 → 사용자 순으로 역순 전파.

---

## 응답 역방향 흐름

Request 는 Step 1 → 6 순서였다면 Response 는 6 → 1 로 올라간다:

```
Step 6: vLLM 이 완성된 JSON 반환
    ↑
Step 5: LiteLLM 이 그대로 forward
    ↑
Step 4: httpx r.json() 이 dict 로 파싱
    ↑
Step 3: chat_guarded 가 hanja_count() 실행
        → 통과: 그대로 반환
        → 실패: 재시도 후 반환
    ↑
Step 2: rag.answer 가 citation 조립
        return (text, citations, model)
    ↑
Step 1: FastAPI 가 QueryResponse 로 감싸서 JSON 응답
    ↑
[사용자] 200 OK + JSON 수신
```

---

## 총 소요 시간 breakdown

전체 5-8초 (질문 짧고 답변 300 tokens 케이스 기준):

| 단계 | 소요 시간 | 비중 |
|---|---|---|
| Step 1 (FastAPI 진입) | ~5ms | 무시 |
| Step 2-1 (Chroma 검색) | 50-200ms | 3% |
| Step 2-2 (프롬프트 조립) | ~1ms | 무시 |
| Step 3 (guard wrapper 오버헤드) | <10ms | 무시 |
| Step 4-5 (HTTP + LiteLLM) | 20-50ms | 1% |
| **Step 6 (Qwen 토큰 생성)** | **3-5초** | **95%+** |

**병목은 압도적으로 LLM 추론**. 최적화 여지는:
- KV 캐시 활용도 높이기 (system prompt 를 앞에 두면 캐시 hit 확률 상승)
- 스트리밍 사용 (사용자 UX 관점에서 첫 토큰까지 시간 단축)
- 더 작은 모델 사용 (7B → 3B) 은 품질과 tradeoff

---

## Phase A 층이 끼어드는 위치

Korean Purity Guard 4개 층 (L1-L4) 이 이 흐름 어디에 붙는지:

| Layer | 이름 | 붙는 위치 |
|---|---|---|
| L1 | 프롬프트 강화 | Step 2-2 (`build_rag_messages` 의 system 메시지) |
| L2 | Temperature 0.1 | Step 4 (`chat` payload) |
| L3 | Post-process 재시도 | Step 3 (`chat_guarded` wrapper) |
| L4 | 모니터링 | 별도 (미구현) |

4개 층 모두 흐름을 방해하지 않고 **각자 다른 지점** 에서 방어. 겹치지 않는 심층 방어.

---

## 스트리밍 경로는 다르다

지금까지 설명은 `POST /rag/query` (비스트리밍) 기준. `POST /v1/chat/completions` 스트리밍은 다음이 다름:

- Step 3 (`chat_guarded`) 를 **거치지 않음** — 중간 재시도 불가능
- Step 4 는 `stream_chat()` 함수 사용 — `AsyncIterator[bytes]` 반환
- 응답이 SSE 청크로 분할되어 실시간 forward

즉 Continue.dev 같은 클라이언트가 스트리밍으로 붙으면 L3 방어가 안 걸린다. 이건 tradeoff (실시간성 vs 완벽한 방어). Phase A 는 RAG 쿼리 경로 (완결성 우선) 위주로 방어를 설계했고, 스트리밍 IDE 자동완성은 원래 한자 leak 확률이 훨씬 낮기 때문에 L1+L2 로 충분하다고 판단.

---

## 정리

- 사용자 질문 하나가 **6개 컴포넌트, 3개 프로세스, 2개 HTTP 홉** 을 통과
- 병목은 압도적으로 Step 6 (LLM 추론). 나머지 다 합쳐도 200ms 미만
- Phase A 방어층 (`chat_guarded`) 은 Step 3 에 wrapper 로 얇게 삽입되어 기존 관심사 분리를 깨지 않음
- 스트리밍 경로는 별도 flow (재시도 불가능하므로)
- 이 구조 덕분에 나중에 vLLM → 다른 백엔드 (OpenAI, Anthropic) 교체 시 `llm.py` 만 손대면 됨

관심사 분리는 그 자체로 목적이 아니라, **바꾸고 싶은 것 하나만 편하게 바꾸기 위한 도구**. 6개 층 구조가 그걸 실증한다.
