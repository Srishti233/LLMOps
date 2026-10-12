"""SQLite request/feedback store (data/monitoring.db). PII is masked before anything is written."""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.pii import mask_pii

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    latency_ms REAL,
    tokens_generated INTEGER,
    model_version TEXT,
    prompt_version TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE,
    ts TEXT NOT NULL,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (ts);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def connect(db_path: str | os.PathLike) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout = 30000")
    return con


def init_db(db_path: str | os.PathLike) -> None:
    with closing(connect(db_path)) as con, con:
        con.executescript(SCHEMA)
    try:  # let the host user and containers (other uids) share the file
        os.chmod(db_path, 0o666)
    except OSError:
        pass


def log_request(db_path, *, request_id: str, prompt: str, response: str, latency_ms: float,
                tokens_generated: int, model_version: str, prompt_version: str, ts: str | None = None) -> None:
    with closing(connect(db_path)) as con, con:
        con.execute(
            "INSERT OR REPLACE INTO requests (request_id, ts, prompt, response, latency_ms, tokens_generated,"
            " model_version, prompt_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (request_id, ts or now_iso(), mask_pii(prompt), mask_pii(response), latency_ms, tokens_generated,
             model_version, prompt_version),
        )


def request_exists(db_path, request_id: str) -> bool:
    with closing(connect(db_path)) as con:
        return con.execute("SELECT 1 FROM requests WHERE request_id = ?", (request_id,)).fetchone() is not None


def log_feedback(db_path, *, request_id: str, rating: str, comment: str | None = None) -> None:
    with closing(connect(db_path)) as con, con:
        con.execute(
            "INSERT INTO feedback (request_id, ts, rating, comment) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(request_id) DO UPDATE SET rating = excluded.rating, comment = excluded.comment, "
            "ts = excluded.ts",
            (request_id, now_iso(), rating, mask_pii(comment) if comment else None),
        )


def recent_prompts(db_path, limit: int = 200) -> list[str]:
    with closing(connect(db_path)) as con:
        rows = con.execute("SELECT prompt FROM requests ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [r["prompt"] for r in rows]


def feedback_counts(db_path) -> dict[str, int]:
    counts = {"up": 0, "down": 0}
    with closing(connect(db_path)) as con:
        for row in con.execute("SELECT rating, COUNT(*) AS n FROM feedback GROUP BY rating"):
            counts[row["rating"]] = row["n"]
    return counts


def feedback_rows(db_path) -> list[dict[str, Any]]:
    """Feedback joined with the request it refers to."""
    with closing(connect(db_path)) as con:
        rows = con.execute(
            "SELECT f.request_id, f.rating, f.comment, f.ts AS feedback_ts, r.prompt, r.response, "
            "r.model_version, r.prompt_version FROM feedback f JOIN requests r USING (request_id) "
            "ORDER BY f.id"
        ).fetchall()
    return [dict(r) for r in rows]


def count_requests(db_path) -> int:
    with closing(connect(db_path)) as con:
        return con.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
