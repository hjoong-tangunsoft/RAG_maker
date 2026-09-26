# LiteLLM은 이미 게이트웨이 아니야? — 게이트웨이는 층위가 있다

> **상태**: seed
> **챕터**: 4. 게이트웨이 층위
> **글감 발견 세션**: 2026-09-27, #37 (LiteLLM 게이트웨이 통합 intent-router) 아키텍처 상담 중

## 오해

"LiteLLM이 이미 프록시 게이트웨이인데, 또 다른 게이트웨이를 앞에 두면 보안이 취약해지는 것 아닌가?"

전제된 가정:
- 게이트웨이는 **하나**여야 한다.
- 프록시는 "가장 바깥에서 유저 요청을 받아 내부로 전달"하는 것이다.
- 그 앞에 뭘 두면 보안 방어선이 뚫린다.

## 이해

**게이트웨이는 층위별로 여러 개 존재하며 판단 기준이 다르다.** 실측 구조:

```
인터넷
  ↓
Apache :443 (llm.tangunsoft.com)          ← 진짜 edge proxy (TLS/도메인/방화벽)
  ├─ /rag/*      → 127.0.0.1:8100 FastAPI
  ├─ /ontology/* → 127.0.0.1:8100 FastAPI
  └─ /*          → 127.0.0.1:4000 LiteLLM  ← "LLM 프록시 게이트웨이" (모델 선택)
                       ↓
                  127.0.0.1:8000 vLLM (Qwen2.5)
```

- **Apache**: edge. 인터넷에 노출. TLS 종료. 가장 앞.
- **LiteLLM**: internal. `127.0.0.1`만 바인딩. **edge가 아니라 LLM 라우터**. 모델 이름 기준.
- **우리 FastAPI(신규 라우터)**: internal. 질문 **의미** 기준으로 plain/rag/ontology 분류.

Intent 라우터를 "LiteLLM 앞에" 둔다는 말은 **Apache 앞이 아니라, Apache 뒤 · LiteLLM 옆**이라는 뜻. 네트워크 노출면은 그대로 Apache.

## 핵심 도식

두 게이트웨이의 판단 기준이 다르다:

| 게이트웨이 | 판단 기준 | 예시 |
|---|---|---|
| LLM 프록시 (LiteLLM) | 모델 이름 | `qwen2.5-7b` vs `mellum-4b` |
| Intent 라우터 (신규 FastAPI) | 질문 의미 | "코딩 질문" vs "우리 문서 검색" vs "계약 DB 조회" |

같은 "게이트웨이"란 이름이지만 하는 일이 다르므로 앞뒤로 쌓는 게 자연스럽다.

## 곁가지

- "프록시 = 무조건 edge"는 오래된 모노리스 시대의 감각. 마이크로서비스에선 layered gateway가 흔함.
- 폐쇄망에서도 같은 원리. edge 보안(Apache)과 internal 라우팅(LiteLLM, FastAPI)의 관심사가 분리되어야 유지보수가 쉽다.

## 확장할 때 다룰 것

- Apache vs Nginx edge proxy 선택 근거
- LiteLLM 관측성(postgres callback)과 라우터 관측성 상관관계 (`router_request_id`)
- Autocomplete 트래픽(mellum-4b) 격리 이유
