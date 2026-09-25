"""FastAPI router exposing ontology objects + actions."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from . import actions, db, seed
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
