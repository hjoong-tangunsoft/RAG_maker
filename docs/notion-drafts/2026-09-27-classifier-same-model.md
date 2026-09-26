# 라우터에 별도 AI 모델 하나 더 필요한가? — 같은 Qwen을 두 번 호출한다

> **상태**: seed
> **챕터**: 5. 라우팅 · Intent · Tool Calling
> **글감 발견 세션**: 2026-09-27, #37 아키텍처 상담 중

## 오해

"라우팅 단에서 AI가 Qwen 말고 하나 더 필요하다는 거지?"

Intent 분류에 LLM tool calling을 쓴다는 말을 듣고, **분류용 별도 모델을 새로 배포해야 한다**고 해석했다. GPU 부담이 두 배가 되는 건 아닌지 우려.

## 이해

**추가 모델 없다. 기존 Qwen을 두 번 호출한다.**

```
1회차 — 분류용 (빠름)
  라우터 → LiteLLM → Qwen
  프롬프트: "이 질문 route_plain/rag/ontology 중 뭐?"
  파라미터: tool_choice=required, temperature=0, max_tokens=20
  응답: tool_calls: [{name: "route_ontology"}]
  → 소요: 100~300ms

2회차 — 답변용 (본체)
  라우터 → /ontology service → LiteLLM → Qwen (+ SQL tool loop)
  프롬프트: "삼성전자 계약 언제 갱신이야?"
  응답: (진짜 답변)
```

**새 모델 배포 X. 새 systemd unit X. 새 GPU 필요 X.**

## 세 옵션 비교

| 옵션 | 별도 모델 | 오버헤드 | 정확도 | 유지보수 |
|---|---|---|---|---|
| B0 키워드 | X | 0ms | 낮음 (semantic 실패) | 룰셋 폭발 |
| B1 소형 분류 LLM | **O (GPU 부담)** | 50~150ms | 높음 | 모델 관리 부담 |
| B2 메인 LLM tool call | X | 100~300ms | 높음 | 프롬프트만 관리 |

폐쇄망에선 B1의 "GPU 하나 더"가 실질적 blocker. B2는 같은 vLLM 인스턴스 재활용이라 배포 비용 0.

## 왜 분류 호출이 그렇게 빠른가

- `max_tokens=20` — tool_call JSON만 나오면 즉시 종료. 문장 생성 없음.
- `temperature=0` — sampling 없이 argmax. 계산 짧음.
- `tool_choice=required` — LLM이 헛소리 못 함. 첫 토큰부터 tool 형식.

7B 모델 기준 30~50 토큰 생성이 대략 200ms 미만. 동시성 확보되면 QPS 저하도 미미.

## 곁가지

- "AI 하나 = GPU 하나"라는 감각은 클라우드 SaaS API (OpenAI, Anthropic) 사용자에 더 강하다. 온프렘 vLLM에선 **모델 하나에 여러 논리적 용도**를 얹는 게 자연스럽다.
- 같은 이유로 온톨로지 에이전트의 SQL tool loop도 별도 모델 없이 Qwen 하나로 돈다.

## 확장할 때 다룰 것

- 분류 호출 실패(LLM이 잘못된 tool 고름) 관측 방법
- Shadow mode로 초기 정확도 측정하는 절차
- 이후 정확도 한계 도달 시 B1으로 upgrade하는 조건
