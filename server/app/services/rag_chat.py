"""RAG chat completion service (extracted from main.py, Issue #37 Step 4).

Full pipeline: tool-call detection, teach trigger, RAG retrieval,
streaming + non-streaming guarded generation, citations footer.

The endpoint POST /rag/v1/chat/completions in main.py is now a thin
wrapper around `rag_chat_service`. The intent router also calls this
function directly on 'rag' route (no internal HTTP self-call).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import llm, rag
from ..config import settings
from ..schemas import ChatCompletionRequest

log = logging.getLogger(__name__)


async def rag_chat_service(body: ChatCompletionRequest, request: Request) -> Any:
    """Full RAG chat completion pipeline.

    Phases:
    - Phase F (Issue #23): detect tool-calling mode; skip RAG when the
      client is orchestrating tools (Continue.dev agent mode).
    - Path 3 teach trigger: detect natural-language save requests
      ("학습해", "저장해", "기억해", "@save") and auto-ingest.
    - RAG retrieval + injection into messages_out.
    - Streaming path: tools → raw stream; text → guarded then fake SSE.
    - Non-streaming path: guarded generation.
    - Citations: attach to response and optionally to body content.
    """
    # ---- Phase F: detect tool-calling mode ----
    has_tools = bool(body.tools)
    has_tool_context = any(
        (m.tool_calls or m.tool_call_id or m.role == "tool")
        for m in body.messages
    )
    should_inject_rag = body.rag and not has_tools and not has_tool_context

    # ---- Path 3: teach trigger detection ----
    if should_inject_rag and body.messages:
        from .. import teach as _teach

        last_user_msg = next(
            (m for m in reversed(body.messages) if m.role == "user"),
            None,
        )
        if last_user_msg:
            trigger = _teach.detect_teach_trigger(last_user_msg.content)
            if trigger:
                try:
                    content, strategy = _teach.extract_teachable_content(
                        current_msg=last_user_msg.content,
                        trigger=trigger,
                        prev_assistant_msg=_teach.find_prev_assistant(body.messages),
                    )
                    result = _teach.auto_ingest(
                        content=content,
                        trigger=trigger,
                        strategy=strategy,
                    )
                    confirmation = _teach.format_confirmation_message(result, content)
                    return JSONResponse({
                        "id": f"chatcmpl-teach-{result['doc_id'][-8:]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": body.model or settings.default_model,
                        "choices": [{
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": confirmation,
                            },
                        }],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        "teach": result,
                    })
                except ValueError as e:
                    log.info("teach trigger detected but extraction failed: %s", e)

    # ---- RAG injection ----
    messages_out: list[dict[str, Any]]
    hits: list[dict[str, Any]] = []
    if should_inject_rag:
        last_user = next(
            (m.content for m in reversed(body.messages)
             if m.role == "user" and m.content),
            None,
        )
        if last_user:
            hits = rag.retrieve(last_user, k=body.rag_k, where=body.rag_filter)
        messages_out = rag.inject_rag_into_chat(body.messages, hits)
    else:
        messages_out = [m.model_dump(exclude_none=True) for m in body.messages]

    # ---- Streaming path ----
    if body.stream:
        if has_tools:
            async def gen_tools():
                async for chunk in llm.stream_chat(
                    messages_out,
                    model=body.model,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens,
                    tools=body.tools,
                    tool_choice=body.tool_choice,
                ):
                    yield chunk
            return StreamingResponse(gen_tools(), media_type="text/event-stream")

        # Phase E-1: buffer through chat_guarded then emit as fake SSE.
        guarded = await llm.chat_guarded(
            messages_out,
            model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
        content = guarded["choices"][0]["message"]["content"]
        model_name = guarded.get("model", body.model or settings.default_model)
        citations_data = None
        if should_inject_rag and hits:
            citations_objs = rag._hits_to_citations(hits)
            citations_data = [c.model_dump() for c in citations_objs]
            if settings.append_citations_to_body:
                content = content + rag.format_citations_footer(citations_objs)

        async def gen():
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            chunk_size = 40
            for i in range(0, len(content), chunk_size):
                piece = content[i:i + chunk_size]
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                await asyncio.sleep(0.02)
            final = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            if citations_data:
                final["citations"] = citations_data
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---- Non-streaming path ----
    resp = await llm.chat_guarded(
        messages_out,
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        tools=body.tools,
        tool_choice=body.tool_choice,
    )
    if should_inject_rag and hits:
        citations_objs = rag._hits_to_citations(hits)
        resp["citations"] = [c.model_dump() for c in citations_objs]
        if settings.append_citations_to_body:
            try:
                resp["choices"][0]["message"]["content"] = (
                    resp["choices"][0]["message"]["content"]
                    + rag.format_citations_footer(citations_objs)
                )
            except (KeyError, IndexError, TypeError):
                log.warning("could not append citations footer: unexpected response shape")
    return JSONResponse(resp)
