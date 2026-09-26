"""FastAPI router exposing ontology objects + actions."""
from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import actions, agent, db, seed
from .models import (
    Contract,
    Customer,
    Product,
    RenewalRiskReport,
    ResponsePlan,
    SupportTicket,
    Vendor,
)

router = APIRouter(prefix="/ontology", tags=["ontology"])


# ---------- admin ----------

@router.get("/health")
def health() -> dict[str, str | int]:
    db.init_db()
    with db.connect() as conn:
        counts = {
            t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
            for t in ("vendor", "product", "customer", "engineer",
                      "contract", "license", "support_ticket")
        }
    return {"status": "ok", "db_path": str(db.db_path()), **counts}


@router.post("/admin/seed")
def admin_seed() -> dict[str, object]:
    rows = seed.load()
    return {"db_path": str(db.db_path()), "rows": rows}


# ---------- object listings ----------

@router.get("/customers")
def list_customers() -> list[Customer]:
    with db.connect() as conn:
        return [Customer(**dict(r)) for r in conn.execute("SELECT * FROM customer")]


@router.get("/vendors")
def list_vendors() -> list[Vendor]:
    with db.connect() as conn:
        return [Vendor(**dict(r)) for r in conn.execute("SELECT * FROM vendor")]


@router.get("/products")
def list_products() -> list[Product]:
    with db.connect() as conn:
        return [Product(**dict(r)) for r in conn.execute("SELECT * FROM product")]


@router.get("/contracts")
def list_contracts() -> list[Contract]:
    with db.connect() as conn:
        return [Contract(**dict(r)) for r in conn.execute("SELECT * FROM contract")]


@router.get("/customers/{customer_id}/tickets")
def list_customer_tickets(customer_id: str) -> list[SupportTicket]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM support_ticket WHERE customer_id=? ORDER BY opened_at DESC",
            (customer_id,),
        ).fetchall()
    if not rows:
        # confirm the customer exists so 404 vs empty is meaningful
        with db.connect() as conn:
            if conn.execute("SELECT 1 FROM customer WHERE id=?", (customer_id,)).fetchone() is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "customer not found")
    return [SupportTicket(**dict(r)) for r in rows]


# ---------- actions ----------

class RenewalRiskRequest(BaseModel):
    days_until_renewal: int = Field(default=30, ge=1, le=365)
    min_recent_tickets: int = Field(default=3, ge=0, le=100)
    recent_window_days: int = Field(default=90, ge=1, le=365)
    vendor_name: str | None = "JetBrains"


@router.post("/actions/renewal-risk")
def action_renewal_risk(body: RenewalRiskRequest) -> RenewalRiskReport:
    return actions.renewal_risk(
        days_until_renewal=body.days_until_renewal,
        min_recent_tickets=body.min_recent_tickets,
        recent_window_days=body.recent_window_days,
        vendor_name=body.vendor_name,
    )


class DraftPlanRequest(BaseModel):
    customer_id: str


@router.post("/actions/draft-response-plan")
def action_draft_plan(body: DraftPlanRequest) -> ResponsePlan:
    try:
        return actions.draft_response_plan(body.customer_id)
    except ValueError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e


# ---------- agent (LLM + tool calling over ontology) ----------

class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    system_prompt: str | None = None
    model: str | None = None
    max_iterations: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=16, le=8192)


@router.post("/agent/chat")
async def agent_chat(body: AgentChatRequest) -> dict:
    return await agent.run(
        user_message=body.message,
        system_prompt=body.system_prompt,
        model=body.model,
        max_iterations=body.max_iterations,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )


# ---------- OpenAI-compatible wrapper (for Continue.dev, OpenAI SDKs, etc.) ----------

class OAIMessage(BaseModel):
    role: str
    content: str | None = None


class OAIChatRequest(BaseModel):
    model: str | None = None
    messages: list[OAIMessage]
    temperature: float | None = 0.2
    max_tokens: int | None = 1024
    stream: bool | None = False


@router.post("/v1/chat/completions")
async def openai_chat_completions(body: OAIChatRequest):
    user_msgs = [m for m in body.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no user message")
    last_user = user_msgs[-1].content or ""
    if not last_user.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty user message")

    system_msg = next((m.content for m in body.messages if m.role == "system"), None)

    real_model = None if (body.model or "").strip() in ("", "ontology-agent") else body.model

    if body.stream:
        return StreamingResponse(
            _stream_agent(body, last_user, system_msg, real_model),
            media_type="text/event-stream",
        )

    trace = await agent.run(
        user_message=last_user,
        system_prompt=system_msg,
        model=real_model,
        temperature=body.temperature if body.temperature is not None else 0.2,
        max_tokens=body.max_tokens if body.max_tokens is not None else 1024,
    )
    answer_text = trace.get("answer", "")
    resp_id = f"chatcmpl-ont-{uuid.uuid4().hex[:16]}"
    resp_model = trace.get("model") or body.model or "qwen2.5-7b"
    created = int(time.time())

    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": created,
        "model": resp_model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": trace.get("answer", "")},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "ontology_trace": {
            "iterations": trace.get("iterations"),
            "tool_calls": trace.get("tool_calls", []),
        },
    }


@router.get("/v1/models")
async def openai_list_models() -> dict:
    return {
        "object": "list",
        "data": [{
            "id": "ontology-agent",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "tangunsoft",
        }],
    }


_DONE = object()


async def _stream_agent(body: OAIChatRequest, user_msg: str, system_msg: str | None, real_model: str | None):
    """SSE generator that shows tool calls in real time as they happen.

    Uses an asyncio.Queue as a bridge: agent.run() executes in a background task
    and pushes events via on_event; this generator drains the queue and yields
    OpenAI-format chunks so Continue.dev renders each step as it arrives.
    """
    resp_id = f"chatcmpl-ont-{uuid.uuid4().hex[:16]}"
    resp_model = body.model or "ontology-agent"
    created = int(time.time())
    queue: asyncio.Queue = asyncio.Queue()

    def chunk(text: str, finish: str | None = None) -> str:
        payload = {
            "id": resp_id, "object": "chat.completion.chunk",
            "created": created, "model": resp_model,
            "choices": [{
                "index": 0,
                "delta": {"content": text} if text else {},
                "finish_reason": finish,
            }],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def on_event(evt: dict) -> None:
        t = evt.get("type")
        if t == "thinking":
            it = evt["iteration"]
            msg = (
                "사용자 질문을 분석하고 있습니다..."
                if it == 1
                else f"이전 결과를 바탕으로 다음 단계를 판단하고 있습니다... (반복 {it})"
            )
            await queue.put(f"\n{msg}\n")
        elif t == "tool_call":
            await queue.put(
                f"\n{evt['narration']}\n\n"
                f"<details><summary>호출 인자 원본</summary>\n\n"
                f"```json\n{evt['arguments']}\n```\n\n</details>\n"
            )
        elif t == "tool_result":
            await queue.put(
                f"\n{evt['narration']}\n\n"
                f"<details><summary>결과 원본</summary>\n\n"
                f"```json\n{evt.get('preview', '')}\n```\n\n</details>\n"
            )

    async def run_agent():
        try:
            trace = await agent.run(
                user_message=user_msg,
                system_prompt=system_msg,
                model=real_model,
                temperature=body.temperature if body.temperature is not None else 0.2,
                max_tokens=body.max_tokens if body.max_tokens is not None else 1024,
                on_event=on_event,
            )
            await queue.put(("FINAL", trace))
        except Exception as e:
            await queue.put(("ERROR", str(e)))
        finally:
            await queue.put(_DONE)

    asyncio.create_task(run_agent())

    yield chunk("<details open>\n<summary>실행 과정 (클릭해서 접기/펼치기)</summary>\n\n")

    final_answer = ""
    while True:
        item = await queue.get()
        if item is _DONE:
            break
        if isinstance(item, tuple):
            kind, payload = item
            if kind == "FINAL":
                final_answer = payload.get("answer", "")
            elif kind == "ERROR":
                yield chunk(f"\n\n[error: {payload}]")
            continue
        yield chunk(item)

    yield chunk("\n</details>\n\n")
    if final_answer:
        yield chunk(final_answer)
    yield chunk("", finish="stop")
    yield "data: [DONE]\n\n"
