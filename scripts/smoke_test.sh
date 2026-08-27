#!/usr/bin/env bash
# End-to-end smoke test for the RAG service.
#
#   ./smoke_test.sh                              # run against local (127.0.0.1:8100)
#   BASE=https://llm.tangunsoft.com ./smoke_test.sh   # run against public Apache
#   API_KEY=xxxx ./smoke_test.sh                 # if X-API-Key auth is enabled
#
# All calls are read-only aside from the ingest/delete pair on 'smoke-doc'.

set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8100}"
CURL=(curl -sS -k)
if [[ -n "${API_KEY:-}" ]]; then
  CURL+=(-H "X-API-Key: $API_KEY")
fi

step() { printf "\n\033[1;34m=== %s ===\033[0m\n" "$*"; }
jq_pp() { python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))"; }

step "1. health"
"${CURL[@]}" "$BASE/rag/health" | jq_pp

step "2. stats (initial)"
"${CURL[@]}" "$BASE/rag/stats" | jq_pp

step "3. ingest text (Korean)"
"${CURL[@]}" -X POST "$BASE/rag/ingest/text" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "smoke-doc",
    "source": "smoke.md",
    "text": "탄군소프트는 서울의 SW 회사입니다. 사내 LLM은 Qwen2.5-7B를 vLLM으로 서빙합니다. RAG는 multilingual-e5-base 임베딩과 Chroma 벡터DB를 씁니다."
  }' | jq_pp

step "4. search"
"${CURL[@]}" -X POST "$BASE/rag/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"어떤 임베딩 모델?","k":3}' | jq_pp

step "5. one-shot RAG query"
"${CURL[@]}" -X POST "$BASE/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"사내 LLM은 어떤 모델을 벡터DB로 씁니까?","k":3,"max_tokens":200}' | jq_pp

step "6. OpenAI-compatible chat (rag=true)"
"${CURL[@]}" -X POST "$BASE/rag/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role":"user","content":"탄군소프트 회사 소개해줘"}],
    "max_tokens": 200,
    "rag": true
  }' | jq_pp

step "7. OpenAI-compatible chat (rag=false, straight passthrough)"
"${CURL[@]}" -X POST "$BASE/rag/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [{"role":"user","content":"say hi in 5 words"}],
    "max_tokens": 30,
    "rag": false
  }' | jq_pp

step "8. cleanup"
"${CURL[@]}" -X DELETE "$BASE/rag/docs/smoke-doc" | jq_pp

step "9. stats (final)"
"${CURL[@]}" "$BASE/rag/stats" | jq_pp

echo -e "\n\033[1;32mall smoke tests passed\033[0m"
