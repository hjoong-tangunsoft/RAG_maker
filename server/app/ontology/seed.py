"""Mockup seed data for JetBrains renewal-risk scenario.

Scenario recap: find customers with a JetBrains contract renewing in <=30 days
AND >=3 support tickets in the last 3 months. Seed data must include at least
one positive hit and a few negatives to validate filtering.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from . import db

TODAY = date.today()
NOW = datetime.now()


VENDORS = [
    ("v_jb", "JetBrains"),
    ("v_gh", "GitHub"),
]

PRODUCTS = [
    ("p_idea", "IntelliJ IDEA Ultimate", "v_jb"),
    ("p_pycharm", "PyCharm Professional", "v_jb"),
    ("p_ghe", "GitHub Enterprise", "v_gh"),
]

CUSTOMERS = [
    ("c_samsung", "삼성전자", "electronics"),
    ("c_lg", "LG전자", "electronics"),
    ("c_kakao", "카카오", "internet"),
    ("c_naver", "네이버", "internet"),
    ("c_hyundai", "현대자동차", "automotive"),
]

ENGINEERS = [
    ("e_jin", "진승민", "jin@tangunsoft.com"),
    ("e_park", "박지훈", "park@tangunsoft.com"),
    ("e_lee", "이수현", "lee@tangunsoft.com"),
]


def _contracts() -> list[tuple]:
    """Returns (id, customer_id, product_id, start, renewal, seats, amount)."""
    return [
        # Samsung: JetBrains IDEA, renews in 15 days -> RENEWAL WINDOW HIT
        ("k_samsung_idea", "c_samsung", "p_idea",
         (TODAY - timedelta(days=350)).isoformat(),
         (TODAY + timedelta(days=15)).isoformat(),
         500, 250_000_000),
        # LG: JetBrains PyCharm, renews in 25 days -> RENEWAL WINDOW HIT
        ("k_lg_pycharm", "c_lg", "p_pycharm",
         (TODAY - timedelta(days=340)).isoformat(),
         (TODAY + timedelta(days=25)).isoformat(),
         120, 60_000_000),
        # Kakao: JetBrains IDEA, renews in 90 days -> outside window
        ("k_kakao_idea", "c_kakao", "p_idea",
         (TODAY - timedelta(days=275)).isoformat(),
         (TODAY + timedelta(days=90)).isoformat(),
         200, 100_000_000),
        # Naver: GitHub Enterprise, renews in 10 days -> in window but NOT JetBrains
        ("k_naver_ghe", "c_naver", "p_ghe",
         (TODAY - timedelta(days=355)).isoformat(),
         (TODAY + timedelta(days=10)).isoformat(),
         800, 400_000_000),
        # Hyundai: JetBrains IDEA, renews in 20 days but LOW ticket activity
        ("k_hyundai_idea", "c_hyundai", "p_idea",
         (TODAY - timedelta(days=345)).isoformat(),
         (TODAY + timedelta(days=20)).isoformat(),
         300, 150_000_000),
    ]


def _licenses() -> list[tuple]:
    return [
        ("l_samsung_idea", "k_samsung_idea", 500, "active"),
        ("l_lg_pycharm", "k_lg_pycharm", 120, "active"),
        ("l_kakao_idea", "k_kakao_idea", 200, "active"),
        ("l_naver_ghe", "k_naver_ghe", 800, "active"),
        ("l_hyundai_idea", "k_hyundai_idea", 300, "active"),
    ]


def _tickets() -> list[tuple]:
    """(id, customer_id, product_id, opened_at, closed_at, severity, assignee, title)."""
    rows: list[tuple] = []

    def add(tid: str, cust: str, prod: str, days_ago: int, sev: str, title: str,
            assignee: str | None = "e_jin", closed_days_ago: int | None = None):
        opened = (NOW - timedelta(days=days_ago)).isoformat()
        closed = (NOW - timedelta(days=closed_days_ago)).isoformat() if closed_days_ago else None
        rows.append((tid, cust, prod, opened, closed, sev, assignee, title))

    # Samsung: 4 tickets in last 90 days -> HIT (>=3)
    add("t_s1", "c_samsung", "p_idea", 80, "high", "라이선스 서버 연결 실패", "e_jin", 75)
    add("t_s2", "c_samsung", "p_idea", 50, "medium", "플러그인 호환성 문의", "e_park", 48)
    add("t_s3", "c_samsung", "p_idea", 30, "high", "인증 서버 인증서 갱신 필요", "e_jin", 28)
    add("t_s4", "c_samsung", "p_idea", 10, "critical", "전체 팀 로그인 불가", "e_jin", None)

    # LG: 3 tickets in last 90 days -> HIT (>=3)
    add("t_l1", "c_lg", "p_pycharm", 70, "medium", "메모리 사용량 이슈", "e_lee", 65)
    add("t_l2", "c_lg", "p_pycharm", 40, "high", "디버거 hang", "e_lee", 35)
    add("t_l3", "c_lg", "p_pycharm", 15, "medium", "코드 완성 속도 저하", "e_park", None)

    # Kakao: 5 tickets - but renewal too far (won't hit)
    for i, days in enumerate([85, 60, 45, 30, 12]):
        add(f"t_k{i}", "c_kakao", "p_idea", days, "medium", f"카카오 이슈 {i}", "e_park",
            None if days < 15 else days - 3)

    # Hyundai: only 1 ticket -> filters out
    add("t_h1", "c_hyundai", "p_idea", 25, "low", "설치 스크립트 문의", "e_lee", 20)

    return rows


def load() -> dict[str, int]:
    """Reset DB and load mock data. Returns row counts per table."""
    db.reset_db()
    counts = {}
    with db.connect() as conn:
        conn.executemany("INSERT INTO vendor(id, name) VALUES (?, ?)", VENDORS)
        counts["vendor"] = len(VENDORS)
        conn.executemany(
            "INSERT INTO product(id, name, vendor_id) VALUES (?, ?, ?)", PRODUCTS
        )
        counts["product"] = len(PRODUCTS)
        conn.executemany(
            "INSERT INTO customer(id, name, industry) VALUES (?, ?, ?)", CUSTOMERS
        )
        counts["customer"] = len(CUSTOMERS)
        conn.executemany(
            "INSERT INTO engineer(id, name, email) VALUES (?, ?, ?)", ENGINEERS
        )
        counts["engineer"] = len(ENGINEERS)
        contracts = _contracts()
        conn.executemany(
            "INSERT INTO contract(id, customer_id, product_id, start_date, "
            "renewal_date, seats, amount_krw) VALUES (?, ?, ?, ?, ?, ?, ?)",
            contracts,
        )
        counts["contract"] = len(contracts)
        licenses = _licenses()
        conn.executemany(
            "INSERT INTO license(id, contract_id, seat_count, status) "
            "VALUES (?, ?, ?, ?)",
            licenses,
        )
        counts["license"] = len(licenses)
        tickets = _tickets()
        conn.executemany(
            "INSERT INTO support_ticket(id, customer_id, product_id, opened_at, "
            "closed_at, severity, assignee_id, title) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            tickets,
        )
        counts["support_ticket"] = len(tickets)
    return counts


if __name__ == "__main__":
    import json
    result = load()
    print(json.dumps({"db": str(db.db_path()), "rows": result}, indent=2))
