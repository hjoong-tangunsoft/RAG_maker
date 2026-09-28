"""Ontology chat service adapter (Issue #37 Step 4).

Adapts a ChatCompletionRequest (router-facing schema) into the ontology
module's own OAIChatRequest and delegates to the existing
`openai_chat_completions` endpoint function. This avoids duplicating the
agent orchestration logic in ontology/router.py.
"""
from __future__ import annotations

import logging
from typing import Any

from ..ontology.router import (
    OAIChatRequest,
    OAIMessage,
    openai_chat_completions,
)
from ..schemas import ChatCompletionRequest

log = logging.getLogger(__name__)

_ONTOLOGY_ROLES = ("system", "user", "assistant")


async def ontology_chat_service(body: ChatCompletionRequest) -> Any:
    """Dispatch a router-classified 'ontology' request to the agent.

    The ontology agent doesn't support tool/tool_call roles in history,
    so those are dropped by the role filter. This matches the classifier
    contract: bypass rules (Step 2) already redirect tool-call history
    to plain-proxy before we reach ontology dispatch.
    """
    ont_body = OAIChatRequest(
        model=body.model,
        messages=[
            OAIMessage(role=m.role, content=m.content or "")
            for m in body.messages
            if m.role in _ONTOLOGY_ROLES
        ],
        temperature=body.temperature if body.temperature is not None else 0.2,
        max_tokens=body.max_tokens if body.max_tokens is not None else 1024,
        stream=body.stream,
    )
    return await openai_chat_completions(ont_body)
