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

from .. import rag
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
    {
        "type": "function",
        "function": {
            "name": "list_customers",
            "description": (
                "List all customers registered in the business ontology, "
                "optionally filtered by a vendor they hold a contract with. "
                "Use this when the user asks how many customers exist, for a "
                "customer roster, or 'which customers use [vendor]'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vendor_name": {
                        "type": "string",
                        "description": "Filter to customers with at least one contract for this vendor (e.g. 'JetBrains'). Omit for all customers.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_customer_contracts",
            "description": (
                "List all contracts for one customer (product, vendor, seats, "
                "amount, renewal date, days until renewal). Use this when the "
                "user asks for a customer's contract list, active licenses, or "
                "'what does [customer] have with us'. Provide EITHER customer_id "
                "('c_samsung') OR customer_name ('삼성전자') - if you only know "
                "the display name, pass it as customer_name and the tool will "
                "resolve it. Do NOT invent id literals like 'samsung_id'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Internal customer id (e.g. 'c_samsung'). Prefer this if known.",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Display name (e.g. '삼성전자'). Use this if you don't know the id.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the internal RAG knowledge base (ingested documents, "
                "JIRA issue exports, wiki pages, PDFs, notes, etc.) by semantic "
                "similarity. Use this WHENEVER the user asks about JIRA issues, "
                "documents, notes, tickets exported to the knowledge base, or "
                "any factual content that is NOT customer/contract/renewal data "
                "in the business ontology. NEVER answer 'I need to check the "
                "JIRA API / web' - always call this tool first. Returns the "
                "top matching passages with source labels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in the user's language. Rewrite it to be specific.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "How many passages to retrieve (default 5, max 20).",
                        "default": 5,
                    },
                },
                "required": ["query"],
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
        if name == "list_customers":
            result = actions.list_customers(vendor_name=args.get("vendor_name") or None)
            return result.model_dump_json()
        if name == "list_customer_contracts":
            cid = args.get("customer_id")
            cname = args.get("customer_name")
            if not cid and not cname:
                return json.dumps({"error": "customer_id or customer_name is required"})
            result = actions.list_customer_contracts(
                customer_id=str(cid) if cid else None,
                customer_name=str(cname) if cname else None,
            )
            return result.model_dump_json()
        if name == "rag_search":
            q = str(args.get("query") or "").strip()
            if not q:
                return json.dumps({"error": "query is required"})
            k = max(1, min(int(args.get("k", 5) or 5), 20))
            hits = rag.retrieve(q, k=k)
            items = [
                {
                    "source": (h.get("metadata") or {}).get("source", "unknown"),
                    "score": round(float(h.get("score", 0.0)), 4),
                    "text": (h.get("text") or "").strip()[:800],
                }
                for h in hits
            ]
            return json.dumps({"query": q, "items": items}, ensure_ascii=False)
        return json.dumps({"error": f"unknown tool: {name}"})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"tool execution failed: {type(e).__name__}: {e}"})
