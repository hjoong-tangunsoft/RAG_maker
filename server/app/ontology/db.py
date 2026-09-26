"""SQLite storage for ontology objects. stdlib only, no new deps.

Schema mirrors models.py. Foreign keys enforced. Idempotent init.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock

from ..config import settings

_DB_PATH = settings.data_dir / "ontology.db"
_lock = Lock()


SCHEMA = """
CREATE TABLE IF NOT EXISTS vendor (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS product (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    vendor_id TEXT NOT NULL REFERENCES vendor(id)
);

CREATE TABLE IF NOT EXISTS customer (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    industry TEXT
);

CREATE TABLE IF NOT EXISTS engineer (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS contract (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customer(id),
    product_id TEXT NOT NULL REFERENCES product(id),
    start_date TEXT NOT NULL,
    renewal_date TEXT NOT NULL,
    seats INTEGER NOT NULL,
    amount_krw INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contract_renewal ON contract(renewal_date);

CREATE TABLE IF NOT EXISTS license (
    id TEXT PRIMARY KEY,
    contract_id TEXT NOT NULL REFERENCES contract(id),
    seat_count INTEGER NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS support_ticket (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customer(id),
    product_id TEXT NOT NULL REFERENCES product(id),
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    severity TEXT NOT NULL,
    assignee_id TEXT REFERENCES engineer(id),
    title TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ticket_customer ON support_ticket(customer_id);
CREATE INDEX IF NOT EXISTS idx_ticket_opened ON support_ticket(opened_at);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, isolation_level=None)  # autocommit
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> Path:
    with _lock, connect() as conn:
        conn.executescript(SCHEMA)
    return _DB_PATH


def reset_db() -> Path:
    with _lock:
        if _DB_PATH.exists():
            _DB_PATH.unlink()
    return init_db()


def db_path() -> Path:
    return _DB_PATH
