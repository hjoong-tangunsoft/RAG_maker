# [Best Practice] LLM 라우터 배포 리스크 감축 — Shadow 와 Canary 롤아웃

**분류**: Best Practice
**대상**: 백엔드 개발자 · SRE
**운영환경**: 운영 (production)
**버전**: v1.0 (Issue #37 P5 v3)
**작성 배경**: 2026-09-28 지능형 Intent Router 서버 배포 실전

## TL;DR

새 기능을 사용자한테 노출시키기 전에 **두 단계로 나눠서 검증** 한다.

1. **Shadow 모드**: 새 로직을 실제로 실행하되 결과는 로그로만 남기고 **사용자한테 나가는 응답은 여전히 옛것** (관측 전용).
2. **Canary 모드**: 새 로직의 결과를 **실제로 소수 사용자한테 노출**. Shadow 로 통과한 새 로직이 실사용 상황에서도 잘 동작하는지 검증.

Shadow 만으로는 실사용 통합 검증이 불가능하고, Shadow 없이 바로 Canary 로 가면 배포 초기 리스크가 크다. 둘 다 필요하다.

## 오해 방지 — Shadow 는 접근 제한이 아니다

"Shadow" 라는 단어가 헷갈리게 만들 수 있는데 정리하면:

| 오해 | 실제 |
|---|---|
| "Shadow = 새 기능을 다른 사람이 못 쓰게 막는 것" | ❌ 접근 제한이 아님. 누구나 새 endpoint 호출 가능 |
| "Shadow = 새 기능이 아예 없는 것처럼 동작" | ⭕ 새 로직은 실행되고 로그도 쌓임. 응답만 옛것 |
| "Shadow = 개발자만 쓰는 debug 모드" | ❌ 프로덕션 트래픽 대상 관측 도구 |

즉 shadow 모드가 켜져 있으면 새 기능이 **의도적으로 비활성** 되어 있고, 사용자는 배포 전과 100% 동일한 UX 를 받는다. 하지만 뒷단에서 새 로직이 병렬로 돌아가면서 로그를 남긴다.

## 배경 — Issue #37 지능형 라우터 사례

Continue.dev 를 Agent 모드로 쓰면서 회사 지식 질문을 던지면 이상하게 답하는 문제가 있었다.

- 질문: `탄군소프트 매출 알려줘`
- 응답: LLM 이 `ls`, `read_file` 같은 filesystem tool 을 호출하고 "우선 디렉토리 확인해볼까요?" 라는 말을 반복

원인은 명확하다. Continue.dev Agent 모드는 매 요청마다 **12개 filesystem/코드 tool** 을 body 에 실어 보낸다. LLM 은 이 tool 목록을 보고 자연스럽게 코딩 어시스턴트 모드로 응답한다. 회사 지식은 어디에도 없다.

근본 해결책은 사용자 질문 앞에 **intent classifier** 를 두어 다음 3종으로 분류하고 각각 다른 서비스로 dispatch 하는 것.

| Route | 처리 | 데이터 소스 |
|---|---|---|
| `plain` | vanilla LLM + Continue.dev tools 통과 | 코딩 작업 그대로 |
| `rag` | RAG 검색 → 사내 문서 chunk 를 context 로 주입 | 사내 매뉴얼 · JIRA · 정책 |
| `ontology` | Business ontology agent 로 dispatch | 고객 · 계약 · 라이선스 DB |

새 endpoint `/router/v1/chat/completions` 를 만들고 Qwen tool-calling 으로 분류 (`route_plain`, `route_rag`, `route_ontology` 3개 tool 중 하나 필수 호출). 아키텍처는 `B (LiteLLM 앞 얇은 FastAPI) + B2 (LLM tool-calling classifier)`.

## Shadow 단계 — 관측만

### 동작 흐름

```
클라이언트 (Continue.dev)
    ↓ POST /router/v1/chat/completions {model: qwen2.5-auto, messages, tools: [...12개]}
Router 진입
    ↓ Bypass 규칙 체크 (autocomplete 모델, tool_call 히스토리, allowlist)
Classifier 실행 (Qwen tool-calling)
    ↓ route ∈ {rag, plain, ontology}
if SHADOW=true:
    로그 라인: rid=... client_model=qwen2.5-auto route=plain bypass=shadow(would_route=rag) classifier_ms=989 total_ms=2745
    응답: 무조건 plain-proxy → LiteLLM → vanilla LLM 응답 (배포 전과 동일)
```

Classifier 는 실행되고 결과는 로그에 남지만, 실제 dispatch (RAG/Ontology) 는 **의도적으로 비활성**. 사용자한테 나가는 응답은 배포 이전과 완전히 동일한 vanilla LLM 응답.

### 관측 지표

- **분류 정확도**: 요청별 classifier 판단 route 가 기대와 일치하는가
- **Classifier latency**: 얼마나 걸리는가 (지연 예산 확인)
- **Bypass 비율**: 어떤 요청들이 classifier 를 건너뛰고 있는가 (설계 의도와 부합하는지)

### 실전 관찰 — Shadow 로 잡은 버그 3개

로그로 감지한 실제 버그들:

**1. 가상 model alias 가 downstream 에 전달됨**
- 증상: 모든 요청 HTTP 500. LiteLLM 로그: `ProxyModelNotFoundError: model=qwen2.5-auto`
- 원인: Router 가 클라이언트에서 받은 가상 alias `qwen2.5-auto` 를 그대로 LiteLLM 에 forward. LiteLLM 은 실제 backend `qwen2.5-7b` 만 알아서 400 반환.
- Fix: `_resolve_model()` 헬퍼로 router 경계에서 가상→실제 변환.

**2. Resolve 순서 오류로 classifier 우회**
- 증상: 모든 요청 `bypass=model=qwen2.5-7b not in allowed_models classifier_ms=0`. 실제 shadow 관측 자체가 불가능.
- 원인: `_resolve_model()` 이 bypass 체크 **전에** 실행되어 `qwen2.5-auto` → `qwen2.5-7b` 로 미리 치환. Bypass Rule 4 ("allowlist 밖 모델") 가 이걸 잡아채서 classifier 를 스킵.
- Fix: resolve 시점을 bypass 체크 · classifier 실행 **후** 로 재배치.

**3. Bypass Rule 2 (`tools_in_request`) 가 Agent 모드 100% 잡아냄**
- 증상: Continue.dev 에서 오는 모든 요청 `bypass=tools_in_request classifier_ms=0`. Classifier 는 실질적으로 죽은 코드.
- 원인: 원 설계는 "body 에 tools 있으면 직행" 이었는데 Continue.dev Agent 모드는 매 요청마다 12개 tool 을 body 에 실어 보냄. 100% 발화.
- Fix: Rule 2 삭제. Rule 3 (tool_call 히스토리) 만 유지. 대신 `_dispatch()` 에서 route=rag/ontology 인 경우 `body.tools=None` 처리 (RAG 는 검색 결과로 답, Ontology 는 자체 tool 사용).

이 3개 버그가 **모두 Shadow 단계에서 감지**됐다. 만약 Shadow 없이 바로 Canary 로 갔으면 사용자한테 500 에러 · 잘못된 응답이 즉시 노출됐을 것.

### Shadow 관찰 결과 (배포 하루차)

- **분류 정확도**: 3/3 정확 (탄군 매출 → rag, 삼성 계약 → ontology, 안녕 → plain)
- **Classifier latency**: 935-1017ms (Qwen tool-calling 1 회 호출 비용)
- **Total latency**: shadow 경로는 classifier + plain-proxy = 약 1700-2800ms

## Canary 단계 — 실사용 노출

Shadow 로 검증된 시점에 `INTENT_ROUTER_SHADOW=false` 로 전환. 이 순간부터:

```
Classifier route 결정
    ↓
route=rag       → rag_chat_service (retrieval + injection + LLM 응답 + citation 반환)
route=ontology  → ontology_chat_service (agent iteration + tool 호출 + trace 반환)
route=plain     → plain-proxy + Continue.dev tools 통과
```

### 노출 범위 제어 — Continue.dev config 로 자연스러운 canary

우리 팀의 canary 방식은 서버 단위 트래픽 %분할이 아니다. 더 단순하고 안전한 접근:

- 서버는 `/router/v1/*` endpoint 를 팀 전체에 노출 (Apache 통과)
- 하지만 **Continue.dev config.yaml 에 `Qwen2.5-Auto (Router)` 항목이 추가된 사람만 이 endpoint 를 씀**
- 실질 canary 대상 = 이 세션 담당자 1인
- 나머지 팀원은 여전히 옛날 `Qwen2.5-7B (Company)` · `Qwen2.5-7B (RAG)` 모델 사용
- 팀 영향 zero

Canary 대상자가 "잘 되네" 라고 확인하면 다음 사람 config 에 추가. 문제 있으면 그 한 사람만 IDE 에서 모델 다시 바꾸면 끝.

### 회귀 검증 (Canary 켰을 때도 기존 경로 안 깨져야)

Router 신설이 기존 `/rag/v1/*`, `/ontology/v1/*` endpoint 를 깨트리면 팀 전체 영향. Canary 전환 전후 반드시 확인:

```bash
curl -sk https://llm.tangunsoft.com/rag/v1/models        # HTTP 200 유지
curl -sk https://llm.tangunsoft.com/ontology/v1/models   # HTTP 200 유지
```

기존 endpoint 는 새 `services/rag_chat.py` · `services/ontology_chat.py` 서비스 함수를 감싸는 **얇은 wrapper** 로 축소되어 있어서 로직 자체는 동일 (Step 4 리팩터). 회귀 위험 최소.

## 왜 Shadow 없이 바로 Canary 가 위험한가 — "조용한 버그" 라는 개념

Shadow 와 Canary 를 나누는 근본 이유는 **버그의 두 가지 성격** 때문이다.

### 시끄러운 버그 vs 조용한 버그

| 종류 | 사용자 눈에 보이는가 | 감지 방법 | 대표 사례 |
|---|---|---|---|
| **시끄러운 버그** | 즉시 보임 (HTTP 500, 응답 이상) | Canary 로도 충분 | 의존성 누락, 문법 에러, 명백한 크래시 |
| **조용한 버그** | 안 보임 (응답은 정상처럼 나옴) | 오직 로그로만 감지 | 잘못된 라우팅, 의도된 분기의 코드 사문화, 무한 fallback |

Canary 는 사용자한테 새 응답을 실제로 노출하니 시끄러운 버그는 쉽게 잡는다. 하지만 조용한 버그는 사용자 응답이 여전히 "정상처럼" 보이니 사용자도 관리자도 이상함을 못 느낀다. 시간이 지나도 아무도 문제 인지 못 함.

Shadow 는 이 조용한 버그를 **자동 로그 관측** 으로 잡는다. 사용자가 눈치채지 못하는 내부 상태 이상 (classifier 가 안 돌고 있음, 특정 규칙이 100% 발화 중, 의도된 dispatch 경로가 사문화됨) 을 로그 통계로 감지한다.

### 앞서 언급한 3개 버그를 이 프레임으로 재분류

**Bug 1 (model alias 미변환)** — **시끄러운 버그**
- 증상: 모든 요청 HTTP 500
- Shadow 든 Canary 든 관리자 IDE 에서 즉시 발견 가능
- Shadow 의 추가 이점: 사용자 노출 전에 잡음

**Bug 2 (Resolve 순서 오류)** — **조용한 버그** ⭐
- 증상: 응답은 정상 (plain-proxy), 하지만 classifier 가 실질적으로 죽어있음
- Canary 만 있었으면 관리자 IDE 에서 "왜 라우팅 안 되지" 라고 뒤늦게 며칠 후 발견
- Shadow 로그의 `bypass=model=qwen2.5-7b not in allowed_models classifier_ms=0` 패턴으로만 감지

**Bug 3 (Bypass Rule 2 과잉 발화)** — **조용한 버그** ⭐
- 증상: 사용자한테 나가는 응답 = "옛날처럼 코드 어시스턴트 모드" (배포 전과 100% 동일)
- Canary 만 있었으면 사용자도 관리자도 응답 자체는 정상이라 이상 못 느낌
- Shadow 로그 통계 `5 bypass=tools_in_request` (100% 발화) 로만 감지

### 왜 조용한 버그가 더 위험한가

- **시끄러운 버그**: 발견 즉시 rollback 필요 = 명확한 대응
- **조용한 버그**: 발견이 늦어짐 = 신규 코드가 "동작 안 하는데도 배포된 채로 방치" = **롤아웃 계획 전체가 무의미해짐**

이번 세션의 Rule 2 버그가 딱 이 케이스였다. 만약 Shadow 없이 Canary 로 바로 갔으면:
- 사용자 응답: "옛날처럼 이상함" (배포 전에도 이랬으니 정상)
- 관리자 인지: "다른 팀원한테 canary 확장해도 되겠네" (사실 classifier 는 돌지도 않고 있는데)
- 몇 주 후: 우연히 로그 뒤지다 "어? classifier_ms=0 이 왜 계속 나오지" 라고 발견

Shadow 는 이 시나리오를 **첫 1시간 안에** 잡는다. 로그 통계만 봐도 `bypass=tools_in_request` 가 100% 라는 이상 패턴이 즉시 드러남.

## Rollback 전략

두 단계 rollback 이 있고 각각 5초 이내에 완료된다.

### 즉시 rollback (Canary → Shadow)

```bash
ssh -i ~/AWS-key/GHES_bastion.pem rocky@52.79.62.107 "
  sudo sed -i 's/^INTENT_ROUTER_SHADOW=false$/INTENT_ROUTER_SHADOW=true/' /upload/rag/rag.env
  sudo systemctl restart rag.service
"
```

Router 는 유지되지만 dispatch 는 다시 관측 전용으로 돌아감. 사용자한테 나가는 응답은 옛것.

### 완전 rollback (Router 전체 비활성)

```bash
ssh -i ~/AWS-key/GHES_bastion.pem rocky@52.79.62.107 "
  sudo sed -i 's/^INTENT_ROUTER_ENABLED=true$/INTENT_ROUTER_ENABLED=false/' /upload/rag/rag.env
  sudo systemctl restart rag.service
"
```

Router 자체가 pass-through 로 동작 (classifier 도 안 돌고 그냥 plain-proxy). Router 코드는 배포되어 있지만 실질적으로 없는 것처럼 동작.

### 개인 rollback (서버 건드리지 않고)

canary 대상자가 이상함을 느끼면 서버 롤백 없이 자기 Continue.dev 에서 `Qwen2.5-Auto (Router)` 대신 `Qwen2.5-7B (RAG)` 선택하면 끝. 다른 canary 대상자한테는 영향 없음.

## 정리

| 단계 | 환경변수 | 사용자 응답 | 검증 목표 |
|---|---|---|---|
| **Shadow** | `SHADOW=true` | 배포 전과 동일 | Classifier 정확도 · 지연 · bypass 규칙 |
| **Canary** | `SHADOW=false` | 새 로직 결과 | Dispatch 통합 · retrieval 품질 · end-to-end 지연 |
| **전체 배포** | Continue.dev default 승격 | 팀 전체 새 로직 | 확장성 · 안정성 |

### 핵심 원칙

- **Shadow 없이 Canary 가면**: 사용자 눈에 안 보이는 버그 (classifier 죽어있음, bypass 규칙 오작동) 를 놓친다
- **Canary 없이 Shadow 로 끝나면**: dispatch 코드 절반이 실사용 검증 없이 남는다
- **둘 다 있어야** 완전한 배포 파이프라인이 완성된다

우리 Issue #37 사례는 Shadow 단계에서 버그 3개를 조기 감지 → 로그 관측만으로 튜닝 → Canary 전환 → 실사용 검증 완료 흐름을 실증했다. 코드 변경 없이 환경변수 1줄 토글로 각 단계를 넘나들 수 있게 설계한 것이 관측 · rollback 비용을 최소화한 핵심.

## 참고

- Issue #37 브랜치: `feat/intent-router-37`
- 관련 커밋: `00e1643` (scaffold) → `bcd8947` (Rule 2 삭제)
- 서버 구성: FastAPI `/router/v1/*` → LiteLLM → vLLM (Qwen2.5-7B-Instruct)
- Classifier: Qwen tool-calling (`temperature=0`, `tool_choice="required"`, `max_tokens=20`)
- 관측 필드: `router_request_id`, `client_model`, `route`, `bypass_reason`, `classifier_latency_ms`, `total_latency_ms`
