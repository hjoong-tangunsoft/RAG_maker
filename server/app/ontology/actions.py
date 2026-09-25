"""Business actions on the ontology.

Actions are the *verbs* of the ontology - what the business actually does with
these objects. Each action returns a strongly-typed pydantic model so downstream
tools (LLM, MCP, Agent) can consume them without dict-guessing.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from . import db
from .models import (
    Contract,
    Customer,
    Product,
    RenewalRiskItem,
    RenewalRiskReport,
    ResponsePlan,
    Vendor,
)


def renewal_risk(
    *,
    days_until_renewal: int = 30,
    min_recent_tickets: int = 3,
    recent_window_days: int = 90,
    vendor_name: str | None = "JetBrains",
) -> RenewalRiskReport:
    """Find customers whose contracts renew soon AND had many recent tickets.

    This is the flagship ontology query - the shape that RAG cannot answer
    because it needs joins across Customer/Contract/Product/Vendor/SupportTicket.
    """
    today = date.today()
    window_end = today + timedelta(days=days_until_renewal)
    ticket_since = datetime.now() - timedelta(days=recent_window_days)

    sql = """
    SELECT
        c.id AS contract_id, c.start_date, c.renewal_date, c.seats, c.amount_krw,
        cu.id AS customer_id, cu.name AS customer_name, cu.industry,
        p.id AS product_id, p.name AS product_name,
        v.id AS vendor_id, v.name AS vendor_name,
        (SELECT COUNT(*) FROM support_ticket t
           WHERE t.customer_id = cu.id
             AND t.product_id = p.id
             AND t.opened_at >= ?) AS ticket_count
    FROM contract c
    JOIN customer cu ON cu.id = c.customer_id
    JOIN product p ON p.id = c.product_id
    JOIN vendor v ON v.id = p.vendor_id
    WHERE c.renewal_date <= ?
      AND c.renewal_date >= ?
    """
    params: list[str] = [ticket_since.isoformat(), window_end.isoformat(), today.isoformat()]
    if vendor_name:
        sql += " AND v.name = ?"
        params.append(vendor_name)
    sql += " ORDER BY c.renewal_date ASC"

    items: list[RenewalRiskItem] = []
    with db.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
        for r in rows:
            if r["ticket_count"] < min_recent_tickets:
                continue
            titles = [
                t["title"] for t in conn.execute(
                    "SELECT title FROM support_ticket "
                    "WHERE customer_id=? AND product_id=? AND opened_at>=? "
                    "ORDER BY opened_at DESC",
                    (r["customer_id"], r["product_id"], ticket_since.isoformat()),
                ).fetchall()
            ]
            renewal = date.fromisoformat(r["renewal_date"])
            items.append(RenewalRiskItem(
                customer=Customer(id=r["customer_id"], name=r["customer_name"], industry=r["industry"]),
                contract=Contract(
                    id=r["contract_id"], customer_id=r["customer_id"], product_id=r["product_id"],
                    start_date=date.fromisoformat(r["start_date"]),
                    renewal_date=renewal, seats=r["seats"], amount_krw=r["amount_krw"],
                ),
                product=Product(id=r["product_id"], name=r["product_name"], vendor_id=r["vendor_id"]),
                vendor=Vendor(id=r["vendor_id"], name=r["vendor_name"]),
                days_until_renewal=(renewal - today).days,
                recent_ticket_count=r["ticket_count"],
                recent_ticket_titles=titles,
            ))

    return RenewalRiskReport(
        generated_at=datetime.now(),
        criteria={
            "days_until_renewal": days_until_renewal,
            "min_recent_tickets": min_recent_tickets,
            "recent_window_days": recent_window_days,
            "vendor": vendor_name or "*",
        },
        items=items,
    )


def draft_response_plan(customer_id: str) -> ResponsePlan:
    """Draft a renewal response plan for one customer.

    v0: rule-based summary from ontology facts. v1 will call the LLM with these
    facts as structured context (this is where RAG-retrieved policy docs join).
    """
    with db.connect() as conn:
        cust = conn.execute("SELECT * FROM customer WHERE id=?", (customer_id,)).fetchone()
        if cust is None:
            raise ValueError(f"customer not found: {customer_id}")

        contracts = conn.execute(
            "SELECT c.*, p.name AS product_name, v.name AS vendor_name "
            "FROM contract c JOIN product p ON p.id=c.product_id "
            "JOIN vendor v ON v.id=p.vendor_id WHERE c.customer_id=?",
            (customer_id,),
        ).fetchall()

        ticket_since = datetime.now() - timedelta(days=90)
        tickets = conn.execute(
            "SELECT * FROM support_ticket WHERE customer_id=? AND opened_at>=? "
            "ORDER BY opened_at DESC",
            (customer_id, ticket_since.isoformat()),
        ).fetchall()

    lines = [
        f"고객: {cust['name']} (산업: {cust['industry'] or '미분류'})",
        f"활성 계약 {len(contracts)}건, 최근 90일 지원 티켓 {len(tickets)}건.",
    ]
    for c in contracts:
        lines.append(
            f"- {c['vendor_name']} {c['product_name']}: {c['seats']}석, "
            f"갱신일 {c['renewal_date']}, {c['amount_krw']:,}원"
        )
    if tickets:
        sev_count: dict[str, int] = {}
        for t in tickets:
            sev_count[t["severity"]] = sev_count.get(t["severity"], 0) + 1
        lines.append(f"- 심각도 분포: {sev_count}")

    actions = [
        "담당 엔지니어 배정 확인 및 미해결 티켓 우선 종료",
        "고객 담당자와 갱신 미팅 2주 내 스케줄",
        "지원 이력 요약 리포트 사전 송부",
        "라이선스 사용률 분석 (실사용 vs 계약 seats)",
    ]
    if len(tickets) >= 3:
        actions.insert(0, "⚠ 최근 지원 이슈 다발 - 근본 원인 분석 리포트 필수")

    return ResponsePlan(
        customer_id=customer_id,
        generated_at=datetime.now(),
        summary="\n".join(lines),
        actions=actions,
    )
