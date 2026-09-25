"""OpenAI function-calling schemas for ontology actions.

These schemas are what we hand to the LLM so it knows *what verbs the business
world has*. The LLM never touches the database; it only decides which tool to
call. Execution stays server-side, in actions.py.

This is the "eyes" concept from the blog: LLM's language capability is fixed
(qwen2.5-7b), but tool schemas give it the ability to *act* on our ontology.
"""
from __future__ import annotations

import json
from typing import Any

from . import actions

# ---------- schemas exposed to the LLM ----------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "renewal_risk",
            "description": (
                "Find customers whose contracts renew soon AND had many recent "
                "support tickets. Use this when the user asks about renewal risk, "
                "upcoming renewals, at-risk customers, or contract expiration "
                "combined with support history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days_until_renewal": {
                        "type": "integer",
                        "description": "Only include contracts renewing within this many days from today.",
                        "default": 30,
                    },
                    "min_recent_tickets": {
                        "type": "integer",
                        "description": "Minimum number of support tickets in the recent window to be considered risky.",
                        "default": 3,
                    },
                    "recent_window_days": {
                        "type": "integer",
                        "description": "How many days back to count support tickets.",
                        "default": 90,
                    },
                    "vendor_name": {
                        "type": "string",
                        "description": "Filter by vendor name (e.g. 'JetBrains', 'GitHub'). Omit for all vendors.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_response_plan",
            "description": (
                "Draft a renewal response plan for one customer. Returns a summary "
                "of their contracts + recent support activity plus recommended "
                "next actions. Use after renewal_risk once a specific customer is "
                "identified, or when the user asks for a plan/proposal for a "
                "named customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer id (e.g. 'c_samsung'). If you only know the customer name, call renewal_risk first to discover ids.",
                    },
                },
                "required": ["customer_id"],
            },
        },
    },
]


# ---------- dispatcher: tool call -> ontology function -> JSON string ----------

def dispatch(name: str, arguments_json: str) -> str:
    """Execute a tool call and return JSON-serialized result for the LLM.

    Errors are returned as JSON `{"error": "..."}` so the LLM can recover.
    """
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid JSON arguments: {e}"})

    try:
        if name == "renewal_risk":
            result = actions.renewal_risk(
                days_until_renewal=int(args.get("days_until_renewal", 30)),
                min_recent_tickets=int(args.get("min_recent_tickets", 3)),
                recent_window_days=int(args.get("recent_window_days", 90)),
                vendor_name=args.get("vendor_name") or None,
            )
            return result.model_dump_json()
        if name == "draft_response_plan":
            cid = args.get("customer_id")
            if not cid:
                return json.dumps({"error": "customer_id is required"})
            result = actions.draft_response_plan(customer_id=str(cid))
            return result.model_dump_json()
        return json.dumps({"error": f"unknown tool: {name}"})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"tool execution failed: {type(e).__name__}: {e}"})
