# BUGLOG

Ontology Agent / RAG 파이프라인에서 발견된 버그를 시간순으로 누적하는 저널.

## 기록 규칙

- **한 버그 = 한 섹션**, 번호는 단조 증가 (`#1`, `#2`, ...). 재현/근본원인 확정 전이라도 발견 즉시 스텁으로 추가한다.
- 각 항목은 아래 6필드 필수. 불명확 시 `TBD`.
  - **증상**: 사용자가 관찰한 표면 현상 (재현 가능한 최소 프롬프트/입력 포함)
  - **재현**: 최소 재현 절차 (프로브 스크립트/시나리오 이름 등 링크)
  - **근본원인**: 코드/설정 어디서, 왜 발생했는지 (파일:라인 인용)
  - **해결**: 실제 코드 변경 요지 + 대안을 배제한 이유
  - **검증**: 어떤 프로브/유닛테스트/로그로 재발 방지를 확인했는지
  - **커밋**: SHA + 브랜치
- 상태 라벨: `resolved` / `mitigated` (근본해결 미완, 회피만 적용) / `open` / `wontfix`
- 우선순위: `P0` (프로덕션 blocker) / `P1` (사용자 관측 가능 결함) / `P2` (로그 소음·개발자 UX)
- 관련 버그는 `Related: #N` 으로 링크.

---

## Index

| # | 제목 | 상태 | 우선순위 | 발견일 | 커밋 |
|---|---|---|---|---|---|
| 1 | Content 채널로 원시 tool_call 리크 | resolved | P0 | 2026-09-26 | `51f138f` |
| 2 | Rewriter가 이전 subject를 concat | resolved | P1 | 2026-09-26 | `51f138f` |
| 3 | Rewriter refusal 형태 출력을 그대로 반환 | resolved | P1 | 2026-09-26 | `a4c471b` |
| 4 | Trace wrapper가 assistant history에 재유입 | resolved | P1 | 2026-09-26 | (이전 세션) |
| 5 | 5턴 이상 대화에서 stale context leakage | resolved | P1 | 2026-09-26 | (이전 세션) |
| 6 | Chroma HNSW 중복 persist WARNING + posthog CRITICAL 소음 | resolved | P2 | 2026-09-26 | `51f138f` |
| 7 | Rewriter 출력 3배 이상 팽창 시 원문 무시 | resolved | P1 | 2026-09-26 | `a4c471b` |
| 8 | 짧은 인사("안녕") 입력 시 무리한 tool 호출 시도 | resolved | P1 | 2026-09-26 | (이전 세션) |
| 9 | 툴 결과 없는 숫자/사실 환각 | resolved | P0 | 2026-09-26 | `51f138f` |
| 10 | `name = fn(...)` Python assignment 형태 리크 | resolved | P0 | 2026-09-26 | `51f138f` |
| 11 | OpenAI wrapper `{"name":..., "arguments":...}` + 노이즈 문자 리크 | resolved | P0 | 2026-09-26 | `51f138f` |
| 12 | 한국어 답변에 중국어(简体) 문자 혼재 | resolved | P0 | 2026-09-26 | `51f138f` |

---

## #1 Content 채널로 원시 tool_call 리크

- **상태**: resolved · **우선순위**: P0 · **발견**: 2026-09-26
- **증상**: 답변 본문에 `rag_search {"query": "..."}` 같은 문자열이 그대로 사용자에게 노출.
- **재현**: `/tmp/rag_probe.py` A/B 시나리오 중 랜덤 재현. Qwen2.5-7B 특성상 `tool_calls` 채널 대신 content로 호출 문자열을 뱉음.
- **근본원인**: `agent.py::run()` 루프가 `choice.tool_calls`만 신뢰. content에 섞인 호출을 실행/치환하지 않고 그대로 `final_text`로 채택.
- **해결**: `_recover_leaked_tool_call()` 안전망 신설 (`agent.py`). content를 스캔해 (a) `name {json}`, (b) `name(kwargs)`, (c) `name = name(...)` (Python assign), (d) OpenAI wrapper `{"name":..., "arguments":...}` 네 패턴을 매칭 → 실제 `tools.dispatch()` 실행 → 결과를 system note로 messages에 주입 → 루프 재-iterate. **대안 (프롬프트만으로 억제) 배제 이유**: 7B 모델은 프롬프트 강제해도 5~10% 확률로 리크. 방어층 필수.
- **검증**: 유닛테스트 9/9 통과. 25/25 프로브 회귀에서 리크 0건.
- **커밋**: `51f138f` (feature/ontology-step-a)

## #2 Rewriter가 이전 subject를 concat

- **상태**: resolved · **우선순위**: P1 · **발견**: 2026-09-26
- **증상**: "JetBrains 갱신 리스크?" → assistant 응답 → "MongoDB는?" 후속 질문 시 rewriter가 `"JetBrains와 MongoDB 갱신 리스크"` 처럼 두 subject를 병합.
- **재현**: 프로브 `F1: 벤더 교체 (JetBrains -> MongoDB)`.
- **근본원인**: `_REWRITE_SYSTEM` 규칙에 "subject 교체 시 REPLACE" 지침이 없어 모델이 안전빵으로 합침.
- **해결**: `_REWRITE_SYSTEM` rule 4 신설 — "When the latest message names a different subject, REPLACE — do NOT concatenate." + 예시 2개.
- **검증**: 프로브 F1/F2/F3 모두 rewritten 문장에 단일 subject만 존재.
- **커밋**: `51f138f`
- **Related**: #3, #7

## #3 Rewriter refusal 형태 출력을 그대로 반환

- **상태**: resolved · **우선순위**: P1 · **발견**: 2026-09-26
- **증상**: Rewriter가 `"I cannot rewrite this."` 같은 메타 응답을 뱉을 때 그대로 사용자 질문으로 사용 → 하위 agent가 완전히 오답.
- **재현**: 프로브 `D3: 이전 assistant refusal (rewriter 오염 유도)`.
- **근본원인**: `_rewrite_query()`가 rewriter 출력 형태 검증 없음.
- **해결**: refusal 시그니처(`"cannot"`, `"unable"`, `"i'm sorry"` 등) 감지 시 원문 그대로 반환하는 폴백 추가.
- **검증**: D3 프로브 통과.
- **커밋**: `a4c471b`

## #4 Trace wrapper가 assistant history에 재유입

- **상태**: resolved · **우선순위**: P1 · **발견**: 2026-09-26
- **증상**: 클라이언트가 이전 turn의 `**실행 과정** > ...` 마크다운 trace 전체를 assistant history로 재전송 → rewriter/agent가 그 텍스트를 실제 assistant 발화로 오인.
- **재현**: 프로브 `D4: trace wrapper가 assistant 히스토리에 포함`.
- **근본원인**: 히스토리 필터가 trace 마커를 sanitize하지 않음.
- **해결**: trace 마커(`**실행 과정**`, `> _...`, `---` 세퍼레이터) 제거 후 최종 답변만 남기는 sanitizer 추가.
- **검증**: D4 프로브 통과. rewritten 문장에 trace 잔재 없음.
- **커밋**: (이전 세션)

## #5 5턴 이상 대화에서 stale context leakage

- **상태**: resolved · **우선순위**: P1 · **발견**: 2026-09-26
- **증상**: 5턴 이상 대화 후 rewriter가 첫 turn의 subject를 재소환.
- **재현**: 프로브 `E3: 5턴 긴 대화 (stale context 유출 검증)`.
- **근본원인**: rewriter가 전체 history를 참조.
- **해결**: rewriter가 참조하는 prior context를 **직전 2턴**으로 제한.
- **검증**: E3 프로브 통과.
- **커밋**: (이전 세션)

## #6 Chroma HNSW 중복 persist WARNING + posthog CRITICAL 소음

- **상태**: resolved · **우선순위**: P2 · **발견**: 2026-09-26
- **증상**: 매 쿼리마다 `Add of existing embedding ID: pool-jira-MAN-0198::00000` 같은 로그 + posthog CRITICAL로 journal 도배.
- **재현**: `journalctl -u rag.service -f` 상태에서 `/rag/search` 호출.
- **근본원인** (2026-09-26 정정):
  - `Add of existing embedding ID`는 `chromadb/segment/impl/vector/local_persistent_hnsw.py:339`의 `logger.warning()` 호출. **ERROR 아님, 데이터 손상 아님.** Chroma가 HNSW 인덱스에 이미 존재하는 embedding ID를 재추가하려 할 때 발생하는 내부 양성 경고 (특정 쿼리 경로에서 정상 재-persist 사이클에 발생).
  - 초기 가설("특정 문서가 ingest 시 이중 persist됨")은 **틀림**. 검증: MAN-0198/0199 삭제 후 재삽입해도 `chunk_count=218` 유지 (doc_count=216 + 정상적으로 2 chunk로 split된 다른 문서 2건). 즉 +2 delta는 데이터 손상이 아닌 정상.
  - posthog는 Chroma↔posthog SDK 버전 mismatch, telemetry 실패.
- **해결**: `main.py`에서 두 로거의 레벨 상향:
  - `chromadb.segment.impl.vector.local_persistent_hnsw` → `ERROR` (WARNING 억제)
  - `chromadb.telemetry.product.posthog` → `CRITICAL`
- **검증**: 5회 `/rag/search` 트리거 후 `journalctl` 필터링 결과 `Add of existing` 0건, `posthog` 0건.
- **커밋**: `51f138f`
- **후기**: 초기 가설 검증을 게을리했음. "특정 doc_id가 로그에 등장" ≠ "그 doc이 손상". 다음 유사 버그에서는 먼저 chroma 소스 로그 레벨/맥락부터 확인.

## #7 Rewriter 출력 3배 이상 팽창 시 원문 무시

- **상태**: resolved · **우선순위**: P1 · **발견**: 2026-09-26
- **증상**: Rewriter가 원문의 3배 이상 긴 문장을 만들어 원 질문의 의도를 희석.
- **재현**: 프로브 `D2: 자립 문장 (pass-through 되어야)`.
- **근본원인**: `_rewrite_query()`가 길이 검증 없음.
- **해결**: `len(text) > max(80, len(last_user) * 3)` 인 경우 원문 그대로 반환.
- **검증**: D2 통과.
- **커밋**: `a4c471b`

## #8 짧은 인사("안녕") 입력 시 무리한 tool 호출 시도

- **상태**: resolved · **우선순위**: P1 · **발견**: 2026-09-26
- **증상**: `"안녕"` 입력만으로 `renewal_risk` 등 툴을 시도 → 무의미한 조회 결과 반환.
- **재현**: 프로브 `H2: 빈 문자열 아닌 공백만`.
- **근본원인**: system prompt에 인사·잡담 처리 지침 부재.
- **해결**: `DEFAULT_SYSTEM_PROMPT`에 "인사/잡담은 툴 호출 없이 간단히 답한다" 지침 추가.
- **검증**: H2 통과.
- **커밋**: (이전 세션)

## #9 툴 결과 없는 숫자/사실 환각

- **상태**: resolved · **우선순위**: P0 · **발견**: 2026-09-26
- **증상**: 툴이 반환하지 않은 숫자(고객 수, 계약 개수 등)를 모델이 임의로 생성.
- **재현**: 프로브 `G1: 고객 전체 몇 개 있어?`, `J1: 잘못된 customer_id`.
- **근본원인**: system prompt에 "툴 결과 밖 사실 금지" 명시 없음.
- **해결**: `DEFAULT_SYSTEM_PROMPT`에 "숫자·사실은 오직 툴 결과에서만. 조회 불가 시 '해당 정보는 현재 도구로 조회할 수 없습니다.'로 정직히 답한다" 추가.
- **검증**: G1/J1 통과. 모델이 임의 숫자 대신 명시적 refusal.
- **커밋**: `51f138f`

## #10 `name = fn(...)` Python assignment 형태 리크

- **상태**: resolved · **우선순위**: P0 · **발견**: 2026-09-26
- **증상**: `samsung_id = renewal_risk(...)` 형태로 content에 리크.
- **재현**: 특정 프로브에서 산발 재현.
- **근본원인**: #1 recovery가 `name(...)`만 매칭, 좌변 assignment 접두 미지원.
- **해결**: recovery 정규식에 `\w+\s*=\s*name(...)` 케이스 추가.
- **검증**: 유닛테스트 추가 통과.
- **커밋**: `51f138f`
- **Related**: #1

## #11 OpenAI wrapper `{"name":..., "arguments":...}` + 노이즈 문자 리크

- **상태**: resolved · **우선순위**: P0 · **발견**: 2026-09-26
- **증상**: content에 `ロン\n{"name":"renewal_risk","arguments":{...}}\n♫` 처럼 non-ASCII 노이즈로 감싸진 OpenAI-style wrapper 리크.
- **재현**: 특정 프로브에서 산발 재현.
- **근본원인**: #1 recovery가 앵커 기반이라 노이즈 문자 앞에서 실패.
- **해결**: `_extract_openai_wrapper()` 신설 — 앵커 없이 문자열 전체를 스캔해 balanced JSON 객체를 우선 추출, 그 다음 tool-name-prefix 스캔 순으로 시도.
- **검증**: 유닛테스트 추가 통과.
- **커밋**: `51f138f`
- **Related**: #1, #10

## #12 한국어 답변에 중국어(简体) 문자 혼재

- **상태**: resolved · **우선순위**: P0 · **발견**: 2026-09-26
- **증상**: 한국어 답변에 `"顾客信息不在..."`, `"具体情况是怎样的"` 같은 중국어 简体 혼재.
- **재현**: 프로브 `I1: 이전 turn 없이 대명사`, `I2: 부정 질문`. Qwen2.5-7B가 컨텍스트 부족 시 중국어로 폴백하는 경향.
- **근본원인**: system prompt의 언어 강제("Korean only")를 모델이 어길 때 방어층 없음.
- **해결**: 
  1. `_has_chinese()` 헬퍼 (`\u4e00-\u9fff` 범위 검출).
  2. 루프 exit 전 검사: 사용자 last message가 한국어인데 candidate에 중국어 혼재 시 **최대 2회** 재작성 지시 (`[system note]`로 한국어 전용 재생성 + 명시적 refusal 템플릿 제공).
  3. 재시도 후에도 남으면 **Han 문자 strip** 폴백.
- **대안 배제**: 프롬프트 강화만으로는 이미 실패한 이력이 있어 방어층 필수. Strip만으로는 의미 손실 위험 → 재시도 우선, strip은 최후.
- **검증**: 25/25 프로브 회귀에서 Chinese 문자 0건.
- **커밋**: `51f138f`

---

## 미결/추적 중

- **모델 의미 한계** (infra 버그 아님, 추적만): F3 두 번 연속 subject 교체 시 wiki 정의로 폴백, MongoDB 등 미등록 벤더에 대해 "위험 없음" 오답. LiteLLM intent-router (#37) 도입 후 재평가.
