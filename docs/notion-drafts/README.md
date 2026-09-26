# Notion Drafts — RAG_maker 구축기 (책 형태 블로그)

RAG_maker 시스템을 구축하며 마주친 문제와 개념 오해→이해 전환 지점을 축적하는 초안 저장소.
장기적으로 **한 권의 책 형태 블로그**로 재구성한다.

## 편집 원칙

- **글감 = 오해→이해 전환 지점**. "당연한 줄 알았는데 아니었던" 지점이 독자에게 가장 큰 가치.
- **초안 상태**: 요지·오해·이해·핵심 도식만 담음. 실제 글은 이후 확장.
- **파일명**: `YYYY-MM-DD-slug.md`
- **연결 이슈**: `[Notion Draft] <제목>` 형식으로 GitHub 이슈 발행. 이슈에는 이 파일 경로만 링크.

## 책 구성 (잠정 목차)

1. **RAG 파이프라인 해부** — 요청 하나가 답변으로 돌아오기까지
2. **환각과 방어** — LLM이 없는 사실을 만들어내는 이유와 3층 방어
3. **검색의 센스** — status 필터, recency boost, metadata 자동 추출
4. **게이트웨이 층위** — LLM 프록시와 Intent 라우터가 다른 이유
5. **라우팅 · Intent · Tool Calling** — 같아 보이는 세 용어의 실제 층위
6. **온톨로지 에이전트** — SQL tool calling으로 구조화 데이터 답변
7. **폐쇄망 운영** — 에어갭 환경의 배포·관측·롤백

## Index

| 파일 | 챕터 | 상태 | 이슈 |
|---|---|---|---|
| [2026-09-03-rag-request-flow.md](2026-09-03-rag-request-flow.md) | 1 | draft | — |
| [2026-09-03-anti-hallucination.md](2026-09-03-anti-hallucination.md) | 2 | draft | — |
| [2026-09-03-smart-jira-retrieval.md](2026-09-03-smart-jira-retrieval.md) | 3 | draft | — |
| [2026-09-03-zero-tolerance-hanja.md](2026-09-03-zero-tolerance-hanja.md) | 2 | draft | — |
| [2026-09-27-litellm-vs-intent-router.md](2026-09-27-litellm-vs-intent-router.md) | 4 | seed | #39 |
| [2026-09-27-routing-vs-intent-vs-toolcalling.md](2026-09-27-routing-vs-intent-vs-toolcalling.md) | 5 | seed | #40 |
| [2026-09-27-classifier-same-model.md](2026-09-27-classifier-same-model.md) | 5 | seed | #41 |

- **draft**: 초안 완성. 편집·확장 대상.
- **seed**: 오해→이해 요지만 기록. 확장 필요.
