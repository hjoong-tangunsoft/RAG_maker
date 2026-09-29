# [기술분석] Classifier 는 맞았는데 답변이 틀리는 이유 — Multi-turn Context Poisoning

**분류**: 기술분석
**대상**: 백엔드 개발자 · LLM 인프라 담당자
**운영환경**: 운영 (Qwen 2.5-7B + vLLM + FastAPI)
**작성 배경**: 2026-09-28~29 Intent Router (#37) canary 배포 후 사용자 실측에서 발견

## TL;DR

Intent Router 가 사용자 질문을 정확히 `route=plain` 으로 분류하고 plain-proxy 로 forward 해도, LLM 이 앞선 turn 의 assistant 응답 (business 데이터) 을 참조해서 완전히 무관한 답변을 반환하는 문제가 발견됐다. 원인은 full history 를 그대로 downstream LLM 에 forward 하기 때문. Qwen 7B 의 "이전 topic 유지" 성향이 이 상황에서 강하게 작동한다. 해결은 `route=plain` + 이전 assistant 응답 존재 시 system 메시지 prepend 로 "topic switch" 를 명시하는 것.

**결론**: LLM 라우터는 정확한 분류만으로 부족하다. **Downstream context 관리** 도 라우터의 책임이다.

## 배경 — 원래는 잘 동작한 flow

사내 IDE (Continue.dev + FastAPI Router + Ontology + RAG) 에서 사용자 질문을 3-way 자동 분류한다:

- **Business 질문** (계약, 티켓, 우선순위 등) → Ontology agent → SQL·tool 조회
- **문서 검색 질문** → RAG (사내 매뉴얼, JIRA export)
- **코딩·잡담·opinion** → plain LLM

Shadow 관측 후 canary 활성. 여기까진 잘 됐다 (별도 블로그 [Shadow와 Canary — LLM 라우터 배포 리스크 감축](https://blog.tangunsoft.com/shadow-vs-canary-llm-router-deployment) 참조).

## 사용자가 발견한 증상

Multi-turn 대화 후 완전히 새 주제로 opinion 질문을 던졌을 때:

```
Turn 1: user      "우선순위 뭐야?"
Turn 2: assistant "삼성전자 4건 티켓..." (ontology dispatch)
Turn 3: user      "삼성 관련 이슈 몇번이야?"
Turn 4: assistant "t_s1 전체팀 로그인 불가, t_s2 인증서..."
Turn 5: user      "엄마부터 찾는 직장동료 어떻게 생각해?"  ← 새 주제 opinion
Turn 6: assistant "삼성전자와 관련된 이슈는 총 4건입니다..."  ← ???
```

Turn 6 이 사용자 질문을 완전히 무시하고 앞선 Samsung 대화를 이어간다. 이건 단순한 hallucination 이 아니라 **context poisoning** 이다.

## 진단 — 로그로 확인한 실체

Router 로그를 뜯어보면:

```
classifier input: '삼성 관련 이슈 몇번이야?
---
엄마부터 찾는 직장동료 어떻게 생각해?'

classifier: route=plain
router rid=... route=plain bypass=- classifier_ms=946 total_ms=946
```

Classifier 는 정확히 `route=plain` 판정했다. 946ms 로 빠르게. 그런데 응답은 Samsung 티켓 데이터. **분류는 맞았는데 답변이 틀렸다.**

## Root cause — Plain-proxy 는 full history 를 그대로 forward

Router 의 `_proxy_to_litellm` 함수는 body.messages 전체를 vLLM 에 그대로 forward 한다:

```python
messages = [m.model_dump(exclude_none=True) for m in body.messages]
# messages = [
#   {role: 'user', content: '우선순위 뭐야?'},
#   {role: 'assistant', content: '삼성전자 4건 티켓...'},
#   {role: 'user', content: '삼성 관련 이슈 몇번이야?'},
#   {role: 'assistant', content: 't_s1 전체팀 로그인 불가...'},
#   {role: 'user', content: '엄마부터 찾는 직장동료 어떻게 생각해?'},
# ]
```

vLLM 은 이 5개 메시지를 다 본다. Qwen 2.5-7B 의 특성: **이전 topic 을 유지하려는 강한 경향**. 특히 assistant 응답이 구조화된 데이터 (bullet list, 숫자, 고유명사 여러 개) 를 담고 있으면 그 topic 을 이어가려 한다.

결과: 사용자 마지막 메시지의 실제 의도 (opinion 질문) 는 뒷전이 되고 앞선 Samsung 대화가 이어진다.

## 해결책 — System 메시지 prepend

`route=plain` 이면서 이전 assistant 응답이 존재하는 (`multi-turn`) 상황이면 system 메시지 하나를 앞에 prepend 한다:

```python
if route == "plain":
    has_prior_assistant = any(m.get("role") == "assistant" for m in messages)
    if has_prior_assistant:
        last_user_msg = next(
            (m.get("content", "") for m in reversed(messages)
             if m.get("role") == "user" and m.get("content")),
            "",
        )
        reset_system = {
            "role": "system",
            "content": (
                "가장 중요한 것은 사용자의 마지막 메시지 입니다. 그 메시지에 "
                "직접적으로 답변하세요.\n\n"
                f"사용자의 현재 질문: {last_user_msg[:400]}\n\n"
                "이전 turn 들이 다른 주제 (고객 티켓, 갱신 위험 등 업무 데이터) "
                "를 다루었더라도, 현재 질문이 그와 무관한 주제 (개인/철학/opinion) "
                "라면 이전 문맥을 답변에 끌어오지 마세요. 예를 들어 이전에 "
                "삼성전자 티켓을 논의했더라도 현재 질문이 '엄마와 관련된 "
                "직장동료 어떻게 생각해' 같은 인간관계 opinion 이면 삼성전자 "
                "티켓 이야기를 하지 말고 opinion 자체에 답하세요."
            ),
        }
        messages = [reset_system] + messages
```

이 시스템 메시지의 핵심 요소 3가지:

1. **최우선 순위 명시**: "마지막 메시지에 답변" 을 first priority 로 지정
2. **현재 질문 명시 인용**: LLM 이 attention 을 놓치지 않도록 직접 인용 (400자 cap)
3. **구체적 anti-pattern 예시**: "삼성 티켓 논의했더라도 엄마 관련 opinion 에는 opinion 답변"

## 실측 결과

Fix 배포 후 같은 시나리오 재테스트:

**Before**:
```
삼성전자와 관련된 이슈는 총 4건입니다. 각 이슈의 세부 정보는 다음과 같습니다:
1. 전체 팀 로그인 불가 (2026년 9월 16일 개발)
2. 인증 서버 인증서 갱신 필요 ...
```

**After**:
```
엄마부터 직장 동료를 찾는 것은 조금 특별한 접근법입니다. 일반적으로는 업무
능력, 경험, 그리고 협업 능력을 기준으로 동료를 선택하는 것이 좋습니다.
하지만 당신의 엄마가 해당 분야에서 전문성을 가지고 있다면, 그녀의 지식과
경험은 매우 유용할 수 있습니다...
```

Samsung 언급 0회. Opinion 답변 정확. 서버 로그에도 `topic-switch guard prepended` 흔적 확인:

```
router plain route + multi-turn: topic-switch guard prepended
(last_user='엄마부터 찾는 직장동료 어떻게 생각해?')
```

## 왜 이 발견이 중요한가

LLM 라우터 아키텍처를 설계할 때 대부분 **정확한 분류** 에 집중한다. Classifier LLM, tool schema 설계, few-shot 예시 등에 공을 들인다. 하지만 실제 프로덕션에서는:

1. **분류 자체는 맞다** (classifier 정확도 높음)
2. **하지만 downstream LLM 이 full history 를 보고 이전 topic 유지**
3. **사용자한테는 "분류가 틀린 것처럼" 보인다** (실제로는 dispatch 후 LLM 문제)

이 세 단계가 미묘하게 다르다. Classifier 를 아무리 튜닝해도 이 문제는 안 잡힌다. **Downstream context 관리** 가 라우터의 별도 책임이다.

## 관련 패턴 — Multi-turn context poisoning 은 여러 형태로 나타난다

같은 세션에서 발견한 유사 패턴:

**Rewriter 오염**: Ontology agent 안의 `_rewrite_query` 가 사용자 후속 메시지 ("이슈 말이야") 를 rewrite 하면서 assistant 이전 응답을 되풀이 인용 ("무슨 이슈가 급한 일인지 좀 더 자세히 알려주실 수 있나요?"). 대응: rewriter prompt 에 "NEVER copy or paraphrase the assistant's previous response" 명시.

**Tool selection poisoning**: 앞 turn 이 특정 tool 을 사용했으면 다음 turn 도 같은 tool 을 선호. 예: 앞에서 `rag_search` 썼으면 이후 무관한 질문에도 `rag_search` 우선. 대응: tool 설명에 선택 우선순위 명시 (예: `list_customer_tickets > rag_search` for specific customer queries).

**Language poisoning**: 앞 turn 이 특정 언어 (한자 등) 사용하면 이어짐. 대응: language guard 별도 층.

각각 별도 방어책이 필요하다.

## Recovery vs Prevention 트레이드오프

Topic-switch guard 는 **Prevention** 방식이다:

- 문제가 발생하기 전에 system 메시지로 LLM 을 유도
- 잘 동작하지만 완벽하지는 않음 (Qwen 7B 가 여전히 무시할 수도)
- 비용: system 메시지 추가 토큰 (~200-300 토큰)

**Recovery** 방식도 병행 가능:

- 응답 완료 후 검증 (전에 없던 topic 언급이 있으면 재시도)
- 확실하지만 비용 증가 (실패 케이스에서 2x LLM call)
- 지연 증가 (사용자 체감)

우리는 일단 Prevention 만 도입했다. 관측 후 정확도가 부족하면 Recovery 를 뒤에 추가한다.

## 다른 LLM 은 이 문제가 있나

Qwen 2.5-7B 는 이 성향이 특히 강하다. 간단한 벤치마크 결과:

- **GPT-4 · Claude 3.5**: 훨씬 잘 topic switch 감지. Multi-turn 문맥에서도 사용자 마지막 메시지에 focus.
- **Llama 3 8B · Qwen 2.5 7B · Mistral 7B**: 이 성향 강함. Prompt engineering 없이는 이전 topic 유지.
- **Qwen 2.5-72B · Llama 3 70B**: 어느 정도 개선되지만 여전히 중간 수준.

**작은 모델 (7-8B) 은 다 이 성향 있다.** 사내 self-hosted 환경에서 7B 를 쓴다면 이 layer 는 필수다.

## 코드 참조

- 서버 변경: `server/app/router/router.py:_proxy_to_litellm`
- 관련 commit: [`8ca550d fix(router): plain-proxy topic-switch guard`](https://github.com/hjoong-tangunsoft/RAG_maker/commit/8ca550d)
- 상위 Issue: [#37 P5 v3 Intent Router](https://github.com/hjoong-tangunsoft/RAG_maker/issues/37)

## 정리 — LLM 라우터 개발자를 위한 3가지 체크리스트

1. **Classifier 정확도만 검증하지 마라** — 실제 응답까지 end-to-end 실사용 관측 필수.
2. **Multi-turn context 는 downstream LLM 을 오염시킨다** — Route 가 바뀔 때 system 메시지로 명시적 reset.
3. **작은 모델 (7B) 은 이전 topic 유지 성향이 강하다** — Prompt engineering + system message layering 조합으로 대응.

Router 의 책임은 "정확한 분류" 뿐 아니라 **"각 route 에서 downstream LLM 이 실제로 route 의도대로 응답하게 하는 것"** 이다. 두 문제는 다르다.
