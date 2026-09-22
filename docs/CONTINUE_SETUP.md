# Continue.dev 팀 온보딩 가이드

> 사내 RAG + Agent 어시스턴트를 IDE (VS Code / JetBrains) 에 연결하는 5분 가이드.
> **Chat 모드**: 회사 문서·Jira 검색. **Agent 모드**: 파일·git·터미널 실행.

## 사전 준비

- 지원 IDE 하나:
  - **VS Code** 1.85 이상
  - **JetBrains** IDE 2024.1 이상 (IntelliJ, PyCharm, WebStorm, DataGrip 등)
- 개인 API 키 (관리자에게 요청, 예: `sk-litellm-xxx...`)
- 네트워크: `llm.tangunsoft.com` 접근 가능

## Step 1: Continue.dev 확장 설치

### VS Code
1. Extensions 탭 (`Ctrl+Shift+X`)
2. "Continue" 검색 → 첫 번째 결과 (`continue.continue`) Install
3. 좌측 사이드바에 Continue 아이콘 (⏵ 화살표 로고) 생김

### JetBrains
1. Settings (`Ctrl+Alt+S`) → Plugins
2. Marketplace → "Continue" 검색 → Install
3. IDE 재시작 → 우측 사이드바에 Continue 패널 추가됨

## Step 2: config.yaml 설정

파일 위치:
- **Windows**: `C:\Users\<사용자>\.continue\config.yaml`
- **macOS/Linux**: `~/.continue/config.yaml`

Continue 사이드바 하단 톱니바퀴 아이콘 → "Open Config" 로도 열 수 있음.

**최소 설정** (팀 표준):

```yaml
name: TangunSoft AI
version: 1.0.0
schema: v1

models:
  - name: Qwen2.5-7B (RAG)
    provider: openai
    model: qwen2.5-7b
    apiBase: https://llm.tangunsoft.com/rag/v1
    apiKey: sk-litellm-...  # ← 관리자한테 받은 개인 키
    capabilities:
      - tool_use          # ← 필수! Agent 모드용
    roles:
      - chat
      - edit
      - apply
    defaultCompletionOptions:
      contextLength: 16384
      maxTokens: 2000
      temperature: 0.2

context:
  - provider: code
  - provider: diff
  - provider: terminal
  - provider: problems
  - provider: folder
  - provider: codebase
```

**핵심 3가지**:
- `apiBase: https://llm.tangunsoft.com/rag/v1` — RAG 통합 엔드포인트 (LiteLLM 직행이 아님)
- `capabilities: [tool_use]` — Agent 모드 활성화 조건 (없으면 tool_calls 안 나감)
- `apiKey` — 개인 키 (팀별 예산·감사 로그 추적됨)

**저장 후**: Continue 사이드바 우상단 새로고침 (`↻`) 클릭 → "Config loaded" 확인.

## Step 3: 첫 대화

### Chat 모드 (기본)

Continue 채팅창 하단 좌측:
```
[Chat ▼]  ← 여기 선택
```

예시 질문:
- `우리 회사 코드 리뷰 컨벤션 알려줘`
- `사내 로깅 정책이 뭐야?`
- `MAN-189 이슈 상세 내용 요약해줘`

→ RAG 가 사내 문서·Jira 검색 후 답변.

### Agent 모드 (파일·터미널 접근)

Continue 채팅창 하단 좌측:
```
[Agent ▼]  ← 여기로 전환
```

예시 요청:
- `이 파일 뭐 하는 함수야?` (현재 편집 중인 파일 자동 인식)
- `프로젝트 구조 보여줘`
- `README.md 열어봐`
- `이 함수 우리 컨벤션에 맞게 리팩토링해줘`

→ LLM 이 적절한 tool (파일 읽기, 검색, 편집) 을 자동 호출.

**Auto Accept 설정** (편의):
- 매번 Apply 클릭 귀찮으면 Continue 설정에서 자동 승인 활성화
- Settings → Agent → "Auto approve tool calls" 켜기
- 또는 config.yaml 에 정책 추가 (Continue 공식 문서 참고)

## 세 모드 사용 매트릭스

| 질문 유형 | 모드 | 예시 |
|---|---|---|
| 회사 정책·컨벤션·Jira 조회 | **Chat** | "우리 로깅 규칙은?" |
| 현재 파일·터미널 상태 확인 | **Agent** | "이 파일 뭐 해?" |
| 컨벤션 검증 (하이브리드) | **Agent** | "이 함수 우리 컨벤션 맞아?" |
| 특정 코드 인라인 수정 | **Edit** | 코드 선택 후 `Ctrl+I` |

## 알려진 이슈·해결

### 응답이 안 돌아옴 (hang)

**원인** (2026-09-22 진단 완료): 대화가 5턴 이상 되면 `max_model_len 8192` 초과 (Issue #27).

**해결**:
1. 채팅창 우상단 `+` 클릭해서 New Chat 시작
2. 짧은 질문 위주, 파일 여러 개 붙이지 않기
3. 서버 max-model-len 확장 대기 (Issue #27)

### Agent 모드에서 명령어를 텍스트로만 뱉음

**원인** (2026-09-22 수정 완료): LLM 이 tool_call syntax 를 코드블록으로 leak 하던 버그.

**해결**: 서버 커밋 `ceaf2b5` 로 수정됨. Continue 재로드 후 재시도.

### Chinese 응답 나옴

**원인** (Phase A/E 로 해결): Qwen 2.5 한자 leak 방어 3층 구축.

**해결**: 자동 처리됨. 발견 시 즉시 관리자 알림.

### 파일을 지어내서 답변 (hallucination)

**원인** (2026-09-22 수정 완료): Rule #7 anti-hallucination 추가.

**해결**: 서버 `d343d12` 로 수정됨. 이제 LLM 이 먼저 tool 로 실제 파일 확인 후 답변.

### API 키 재발급 필요

관리자 (`@hjoong-tangunsoft`) 에게 요청. LiteLLM admin UI 에서 즉시 발급 가능.

## 팀 내 활용 예시

### 백엔드 개발자
- "이 API 우리 응답 스키마 규칙 지켜?" (Agent + RAG hybrid)
- "지난 주 Jira 이슈 요약해줘" (Chat, RAG)
- "이 에러 로그 원인 뭐야?" (Agent, currentFile)

### 프론트엔드 개발자
- "우리 팀 React 컨벤션?" (Chat, RAG)
- "이 컴포넌트 리팩토링해줘" (Edit)
- "테스트 파일 만들어줘" (Agent)

### 데이터 엔지니어
- "우리 DAG 네이밍 규칙?" (Chat, RAG)
- "이 SQL 쿼리 최적화해줘" (Edit)
- "데이터 파이프라인 관련 이슈 검색" (Chat, RAG)

## 관리자 용 (참고)

### 개인 API 키 발급 (관리자만)

```bash
curl -X POST https://llm.tangunsoft.com/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "hjoong@tangunsoft.com",
    "team_id": "backend",
    "max_budget": 100.0,
    "duration": "30d",
    "metadata": {
      "spend_logs_metadata": {
        "role": "developer",
        "onboarded_at": "2026-09-22"
      }
    }
  }'
```

응답 예:
```json
{"key": "sk-litellm-...", "user_id": "...", "expires": "..."}
```

이 `key` 를 팀원에게 안전하게 전달. 팀원은 `config.yaml` 의 `apiKey` 자리에 넣음.

### 사용량 조회

- **UI**: `https://llm.tangunsoft.com/litellm-ui/` (예정, Issue #28 Phase 2 완료 후)
- **CLI**: `/global/spend/report`, `/user/daily/activity` API

관련 이슈:
- #28 (LiteLLM 감사·사용량 관리 활성화)
- #23 (Continue.dev 통합 Epic)

## 관련 문서

- [Continue 공식 문서](https://docs.continue.dev/)
- [Continue Agent 모드 가이드](https://docs.continue.dev/features/agent)
- [내부 RAG 설계 원칙](../docs/DESIGN_PRINCIPLES.md)
- [사내 아키텍처](../docs/ARCHITECTURE.md)

## 도움 필요

- **설치·설정 문제**: 관리자 (`@hjoong-tangunsoft`) 에게 문의
- **버그·개선 제안**: [GitHub Issues](https://github.com/hjoong-tangunsoft/RAG_maker/issues)
- **API 키 발급·재발급**: 관리자에게 이메일
