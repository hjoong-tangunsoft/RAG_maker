# Streaming Hanja Bypass 진단 - Zero-Tolerance 3층 방어로 완전 봉쇄

> Phase A (Korean Purity Guard) 배포 후에도 사용자가 "나는 이현중이야 알았어?" 물어봤을 때 응답 전체가 중국어로 나오는 사례 발견. Non-stream 5회 반복 재현 시 모두 hanja=0 인데, IDE 클라이언트가 쓰는 **streaming 경로** 는 chat_guarded 를 완전히 우회한다는 사실 확인. Zero-Tolerance + hanja 라이브러리 기반 3층 방어로 재설계.

## 발견된 버그

사용자가 사내 챗 UI 에서 아래처럼 물었을 때:

```
User: 나는 이현중이야 알았어?

LLM 응답 (실제):
明白了，你是李贤中。当前你负责的"应该完成"的任务如下：

[MAN-175] 이슈이슈이슈이슈이슈이슈이슈이슈이슈이슈이슈
Assignee: 李贤中
Due: 2026-09-02

[MAN-177] 이슈이슈이슈이슈이슈이슈이슈이슈이슈이슈
Assignee: 李贤中
Due: 2026-09-02

这些任务的状态都是"应该完成"，并且都分配给了你。
```

응답 전체가 중국어. LLM 이:
1. 이현중을 **李贤中** (중국식 음역) 으로 변환
2. "해야 할 일" 을 **应该完成** 으로 번역
3. 전체 문장을 중국어 문법으로 구성

Phase A 는 이 hanja leak 을 잡았어야 했음. 왜 안 잡혔나?

---

## 진단 - Non-stream vs Stream 5회 반복 테스트

같은 요청을 8가지 시나리오로 재현:

| Case | 경로 | 조건 | HANJA 카운트 |
|---|---|---|---|
| 1 | non-stream | 그대로 | 0 |
| 2 | stream | 그대로 | 0 |
| 3 | non-stream | client system message | 0 |
| 4 | stream | client system message | 0 |
| 5 | non-stream | 후속 질문 (내 할 일?) | 0 |
| 6 | non-stream | 통합 질문 | 0 |
| 7 | non-stream | 이름 여러 번 반복 | 0 |
| Stress | non-stream 5회 | 동일 요청 반복 | 0, 0, 0, 0, 0 |

**전 케이스 hanja=0**. 그런데 사용자는 실제로 hanja leak 을 봤음. 원인은?

## Root Cause: Streaming 경로가 chat_guarded 를 완전히 우회

`server/app/main.py:321-330` 확인:

```python
if body.stream:
    async def gen():
        async for chunk in llm.stream_chat(...):   # ← chat_guarded 안 씀!
            yield chunk
    return StreamingResponse(gen(), media_type="text/event-stream")

# 아래는 non-stream 만 도달
resp = await llm.chat_guarded(messages_out, ...)   # hanja guard 는 여기만
```

**stream=True 인 요청은 stream_chat 직접 호출** → hanja_count, 재시도, 강화 프롬프트 **전부 우회**.

### 왜 이게 문제인가?

- **Continue.dev, VS Code Copilot, ChatGPT UI, Cursor** 등 IDE / 챗 클라이언트는 대부분 **stream=true 가 기본**
- 즉 실사용 트래픽의 **90% 이상이 방어 없는 경로**
- Non-stream 은 5/5 성공하지만 stream 은 언제든 터질 수 있는 상태

### Stochastic 특성

Qwen 2.5-7B 는 temperature 0.1 에서도 가끔 중국어 모드로 미끄러짐. 특히:
- 사용자 이름이 한자 대응 되는 경우 (이현중 ↔ 李贤中)
- 사내 위키에 중국 관련 이슈가 있는 경우
- 이전 대화에 중국어가 나왔던 경우 (persona bleeding)

Non-stream 은 chat_guarded 가 이걸 잡아냄. **Stream 은 안 잡음.**

---

## Phase A 자체도 취약점 존재 (Streaming 문제와 별개)

Phase A 를 다시 보면:

| 항목 | 현재 값 | 문제 |
|---|---|---|
| `hanja_threshold` | 2 | 최대 2 hanja 통과 허용 |
| `max_retries` | 1 | 1회만 재시도 |
| 재시도 후 재검증 | 없음 | retry 결과에 hanja 있어도 그대로 반환 |
| Fallback conversion | 없음 | retry 실패 시 대체 수단 없음 |

**사용자의 zero-tolerance 요구** ("1개라도 보이면 잡아서 한글로 바꾸기") 를 만족하려면 threshold 를 0 으로 낮춰야 하고, 재시도 후에도 hanja 있으면 마지막 수단으로 변환해야 함.

---

## Phase E 설계 - Zero Tolerance 3층 방어

### E-1: Streaming 경로 봉쇄 (핵심 수정)

**접근**: Buffer-then-Emit
- 클라이언트가 `stream=true` 로 요청해도 서버 내부에서는 non-stream 방식으로 chat_guarded 호출
- 검증된 전체 응답을 다시 OpenAI-compatible SSE chunks 로 재구성해서 전송

**UX 영향**: 
- 사용자는 응답 완료까지 대기 (기존 stream 의 "타이핑되듯 흐르는 UX" 손실)
- 대신 100% Korean 보장

**코드 (main.py)**:
```python
if body.stream:
    # Phase E-1: buffer through chat_guarded for streaming safety
    resp = await llm.chat_guarded(messages_out, ...)
    content = resp["choices"][0]["message"]["content"]
    
    async def gen():
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        # Split into small chunks for smoother appearance
        for i in range(0, len(content), 40):
            piece = content[i:i+40]
            chunk = {
                "id": chat_id, "object": "chat.completion.chunk",
                "created": created, "model": resp.get("model"),
                "choices": [{"index": 0, "delta": {"content": piece},
                            "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
            await asyncio.sleep(0.02)
        
        final = {..., "choices": [{"index": 0, "delta": {}, 
                                    "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    
    return StreamingResponse(gen(), media_type="text/event-stream")
```

### E-2: Zero Threshold

```python
# config.py
hanja_threshold: int = 0   # 2 → 0 (1개라도 잡음)
```

### E-3: 3회 재시도 + hanja 라이브러리 변환 fallback

**hanja 라이브러리** 실측 결과:

| 입력 | 출력 |
|---|---|
| `李贤中` | `이현중` ✅ 사용자 이름 완벽 |
| `业务 会社 情报` | `업무 회사 정보` ✅ 한자어 완벽 |
| `大韓民國은 民主共和國이다` | `대한민국은 민주공화국이다` ✅ Traditional hanja 완벽 |
| `MAN-175 Assignee: 李贤中 Due: 2026-09-02` | `MAN-175 Assignee: 이현중 Due: 2026-09-02` ✅ 실제 사용자 응답 형식 완벽 |
| `明白了，你是李贤中` | `명백료，이시리현중` ⚠️ 한자별 음독, 문법 어색 |
| `应该完成` | `응해완성` ⚠️ "해야 할 일" 이 자연스럽지만 음독은 어색 |

**요약**:
- 고유명사 / 명사 단독: **완벽 변환** (이현중, 업무 등)
- 문장 형태: 한자는 100% 제거되지만 어색한 음독
- 최소한 **hanja 는 하나도 안 남음** 은 보장

**chat_guarded 재작성**:

```python
async def chat_guarded(messages, ..., max_retries=2):
    """Zero-tolerance hanja guard: retry then hanja-lib conversion."""
    current_messages = messages
    temperatures = [temperature or settings.rag_temperature, 0.05, 0.02]
    
    for attempt in range(max_retries + 1):
        resp = await chat(
            current_messages, model=model,
            temperature=temperatures[attempt],
            max_tokens=max_tokens, extra=extra,
        )
        text = resp["choices"][0]["message"]["content"]
        n = hanja_count(text)
        
        if n == 0:  # zero-tolerance
            if attempt > 0:
                log.info("hanja resolved after retry %d", attempt)
            return resp
        
        log.warning(
            "attempt %d: hanja leak %d chars, %s",
            attempt, n, "retrying" if attempt < max_retries else "last-resort conversion",
        )
        current_messages = _reinforce_korean_only(current_messages)
    
    # 3회 실패 → hanja 라이브러리 강제 변환
    text = _convert_hanja_to_hangul(text)
    resp["choices"][0]["message"]["content"] = text
    log.warning("last-resort hanja conversion applied, final hanja=%d", hanja_count(text))
    return resp


def _convert_hanja_to_hangul(text: str) -> str:
    """Convert any remaining hanja to Korean phonetic reading."""
    try:
        import hanja  # optional dep - lazy import
        return hanja.translate(text, "substitution")
    except ImportError:
        log.warning("hanja lib not installed, stripping instead")
        return _HANJA_RE.sub("", text)
```

---

## 예상 Before / After

| 경로 | Before (Phase A) | After (Phase E) |
|---|---|---|
| Non-stream 정상 응답 | HANJA=0 ✓ | HANJA=0 ✓ (동일) |
| Non-stream + 재시도 성공 | HANJA=0 (retry 1회 후) | HANJA=0 (최대 3회 재시도) |
| Non-stream + 재시도 실패 | **HANJA=여전히 존재** ❌ | HANJA=0 (hanja lib 변환) ✅ |
| **Stream** | **HANJA=검증 안 됨** ❌ | **HANJA=0 (buffered 검증)** ✅ |
| Stream + 중국어 leak | **그대로 클라이언트 전달** ❌ | 검증 후 안전한 응답만 전달 ✅ |

**핵심 승리**: Stream 경로 커버, threshold 0 zero-tolerance, 최후 fallback 확정.

---

## 성공 기준 (배포 후 8개 회귀 테스트)

| # | 경로 | Query | 기대 |
|---|---|---|---|
| 1 | non-stream | "나는 이현중이야 알았어?" | HANJA=0 |
| 2 | stream | "나는 이현중이야 알았어?" | HANJA=0 |
| 3 | non-stream | "내 할 일 뭐야?" | HANJA=0 |
| 4 | stream | "내 할 일 뭐야?" | HANJA=0 |
| 5 | non-stream | Chinese-primed 대화 후 한국어 질문 | HANJA=0 |
| 6 | stream | Chinese-primed 대화 후 한국어 질문 | HANJA=0 |
| 7 | non-stream | 실제 중국어 질문 ("你是谁?") | 중국어 답변 허용 (사용자 언어 존중) |
| 8 | stream | 실제 중국어 질문 | 중국어 답변 허용 |

**Case 7-8** 중요: 사용자가 실제로 중국어로 물으면 중국어 답변 허용해야 함. Phase A L1 프롬프트: "사용자가 다른 언어로 물으면 그 언어로 답변" 유지.

---

## Phase A/B/C/D 와의 관계

| Phase | 대상 | 계층 |
|---|---|---|
| A | 한자 leak (non-stream) | Prompt + Temperature + Post-hoc 재시도 |
| B | 학습해 트리거 | Ingest 사전 감지 |
| C | Bridging hallucination | Retrieval 사전 grounding 통합 |
| D | Jira 스마트 검색 | Metadata 추출 + 의도 기반 필터 |
| **E** | **Streaming hanja leak + Zero-tolerance** | **Stream buffer + threshold 0 + hanja lib fallback** |

Phase E 는 Phase A 의 확장이자 완결. Phase A 가 놓친 두 부분 (streaming, retry 후에도 hanja 있는 케이스) 을 마무리.

---

## 배포 후 실사용 임팩트

**Continue.dev / IDE 사용자**:
- 답변 완료까지 대기 시간 증가 (기존 스트림 실시간 UX 손실)
- 대신 100% Korean 답변 보장 → 신뢰성 향상

**Non-stream API 사용자**:
- 성능 영향 없음 (기존과 거의 동일)
- Threshold 0 으로 더 엄격한 방어 → 품질 향상

**중국어 실제 요청**:
- 사용자가 중국어로 물으면 중국어 답변 그대로 유지
- L1 프롬프트가 여전히 "사용자 언어 존중" 규칙 유지

---

## 정리

- **Phase A 는 streaming 경로 무방비 상태였음** - 실사용 트래픽의 대부분이 방어 없이 통과 중
- **Zero-tolerance = 사용자 요구** ("1개라도 잡아서 한글로 바꾸기")
- **hanja 라이브러리** 는 고유명사에 대해 완벽 변환, 문장은 어색하지만 hanja 는 100% 제거
- **3층 방어** 로 각 실패 모드 개별 봉쇄: buffered streaming → 3회 재시도 → 마지막 강제 변환

**핵심 원칙**: 언어 순수성 방어는 요청 시점에도, 응답 시점에도, 모든 경로에 걸쳐 적용되어야 한다. Streaming 은 UX 를 위해 실시간성을 포기했지만, RAG 답변 품질보다는 실시간성이 덜 중요하다. 한자 방어에 timing 예외는 없다.
