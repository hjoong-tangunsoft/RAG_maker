"""Ontology object types (TBox) + pydantic schemas.

Design principle: parse-don't-validate. Every object crossing an API boundary
is a strongly-typed pydantic model. No dicts flowing around.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class LicenseStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"


class TicketSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Vendor(BaseModel):
    id: str
    name: str  # "JetBrains", "GitHub"


class Product(BaseModel):
    id: str
    name: str  # "IntelliJ IDEA Ultimate", "GitHub Enterprise"
    vendor_id: str


class Customer(BaseModel):
    id: str
    name: str  # "삼성전자"
    industry: str | None = None


class Engineer(BaseModel):
    id: str
    name: str
    email: str


class Contract(BaseModel):
    id: str
    customer_id: str
    product_id: str
    start_date: date
    renewal_date: date
    seats: int = Field(ge=1)
    amount_krw: int = Field(ge=0)


class License(BaseModel):
    id: str
    contract_id: str
    seat_count: int = Field(ge=1)
    status: LicenseStatus = LicenseStatus.ACTIVE


class SupportTicket(BaseModel):
    id: str
    customer_id: str
    product_id: str
    opened_at: datetime
    closed_at: datetime | None = None
    severity: TicketSeverity
    assignee_id: str | None = None  # Engineer.id
    title: str


# ---------- Action outputs ----------


class RenewalRiskItem(BaseModel):
    customer: Customer
    contract: Contract
    product: Product
    vendor: Vendor
    days_until_renewal: int
    recent_ticket_count: int
    recent_ticket_titles: list[str]


class RenewalRiskReport(BaseModel):
    generated_at: datetime
    criteria: dict[str, int | str]
    items: list[RenewalRiskItem]


class ResponsePlan(BaseModel):
    customer_id: str
    generated_at: datetime
    summary: str
    actions: list[str]


class CustomerListItem(BaseModel):
    customer: Customer
    contract_count: int
    active_vendors: list[str]


class CustomerListReport(BaseModel):
    generated_at: datetime
    filter: dict[str, str]
    items: list[CustomerListItem]


class ContractDetail(BaseModel):
    contract: Contract
    product: Product
    vendor: Vendor
    days_until_renewal: int


class CustomerContractsReport(BaseModel):
    customer: Customer
    generated_at: datetime
    items: list[ContractDetail]
