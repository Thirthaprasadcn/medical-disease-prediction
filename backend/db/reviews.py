"""Phase 5: persist clinician accept/reject/sign-off in SQLite."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "storage" / "reviews.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS finding_reviews (
                scan_id TEXT NOT NULL,
                finding_id INTEGER NOT NULL,
                decision TEXT NOT NULL CHECK(decision IN ('accept', 'reject')),
                comment TEXT,
                reviewer TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (scan_id, finding_id)
            );
            CREATE TABLE IF NOT EXISTS report_signoff (
                scan_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN ('draft', 'signed')),
                reviewer TEXT,
                signed_at TEXT,
                note TEXT
            );
            """
        )


def _now() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def upsert_finding_review(
    scan_id: str,
    finding_id: int,
    decision: str,
    *,
    reviewer: str = "anonymous",
    comment: str | None = None,
) -> dict[str, Any]:
    if decision not in {"accept", "reject"}:
        raise ValueError("decision must be accept or reject")
    init_db()
    ts = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO finding_reviews (scan_id, finding_id, decision, comment, reviewer, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(scan_id, finding_id) DO UPDATE SET
                decision=excluded.decision,
                comment=excluded.comment,
                reviewer=excluded.reviewer,
                updated_at=excluded.updated_at
            """,
            (scan_id, finding_id, decision, comment, reviewer, ts),
        )
    return {
        "scan_id": scan_id,
        "finding_id": finding_id,
        "decision": decision,
        "comment": comment,
        "reviewer": reviewer,
        "updated_at": ts,
    }


def set_signoff(
    scan_id: str,
    status: str,
    *,
    reviewer: str = "anonymous",
    note: str | None = None,
) -> dict[str, Any]:
    if status not in {"draft", "signed"}:
        raise ValueError("status must be draft or signed")
    init_db()
    ts = _now() if status == "signed" else None
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO report_signoff (scan_id, status, reviewer, signed_at, note)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(scan_id) DO UPDATE SET
                status=excluded.status,
                reviewer=excluded.reviewer,
                signed_at=excluded.signed_at,
                note=excluded.note
            """,
            (scan_id, status, reviewer, ts, note),
        )
    return {
        "scan_id": scan_id,
        "status": status,
        "reviewer": reviewer,
        "signed_at": ts,
        "note": note,
    }


def get_report_reviews(scan_id: str) -> dict[str, Any]:
    init_db()
    with _connect() as conn:
        findings = [
            dict(row)
            for row in conn.execute(
                "SELECT finding_id, decision, comment, reviewer, updated_at FROM finding_reviews WHERE scan_id=?",
                (scan_id,),
            )
        ]
        sign = conn.execute(
            "SELECT status, reviewer, signed_at, note FROM report_signoff WHERE scan_id=?",
            (scan_id,),
        ).fetchone()
    return {
        "scan_id": scan_id,
        "findings": findings,
        "signoff": dict(sign) if sign else {"status": "draft"},
    }


def export_feedback() -> list[dict[str, Any]]:
    """Export accept/reject labels for future retraining."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT scan_id, finding_id, decision, comment, reviewer, updated_at
            FROM finding_reviews
            ORDER BY updated_at ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def export_feedback_jsonl(path: Path | None = None) -> Path:
    path = path or (DB_PATH.parent / "feedback_export.jsonl")
    rows = export_feedback()
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path
