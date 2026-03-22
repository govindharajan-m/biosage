"""
BioSage — Persistence Layer
SQLite-backed storage for workspaces, analyses, and API keys.

Upgradeable to PostgreSQL by swapping the connection string.
Schema is kept intentionally simple — analysis results are stored
as JSON blobs so the schema never needs migration for new result types.
"""

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "biosage.db"


# ── Connection ──────────────────────────────────────────────────────────────

@contextmanager
def _conn():
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Schema ──────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT DEFAULT '',
    color       TEXT DEFAULT '#3B82F6',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    type          TEXT NOT NULL,   -- variant | vcf | literature | gene | chat
    name          TEXT NOT NULL,
    query         TEXT NOT NULL,
    result_json   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'complete',  -- pending | running | complete | error
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    key_hash   TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    last_used  TEXT
);

CREATE INDEX IF NOT EXISTS idx_analyses_workspace ON analyses(workspace_id);
CREATE INDEX IF NOT EXISTS idx_analyses_type ON analyses(type);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
"""

DEFAULT_WORKSPACE_ID = "default"


def init_db():
    """Create tables and seed default workspace."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        con.executescript(SCHEMA)
        # Seed default workspace if missing
        exists = con.execute(
            "SELECT 1 FROM workspaces WHERE id = ?", (DEFAULT_WORKSPACE_ID,)
        ).fetchone()
        if not exists:
            now = _now()
            con.execute(
                "INSERT INTO workspaces VALUES (?,?,?,?,?,?)",
                (DEFAULT_WORKSPACE_ID, "My Workspace", "Default research workspace", "#3B82F6", now, now),
            )
    logger.info(f"Database ready at {DB_PATH}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


# ── Workspaces ──────────────────────────────────────────────────────────────

def list_workspaces() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            """SELECT w.*, COUNT(a.id) as analysis_count
               FROM workspaces w
               LEFT JOIN analyses a ON a.workspace_id = w.id
               GROUP BY w.id
               ORDER BY w.updated_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def create_workspace(name: str, description: str = "", color: str = "#3B82F6") -> dict:
    now = _now()
    uid = _uid()
    with _conn() as con:
        con.execute(
            "INSERT INTO workspaces VALUES (?,?,?,?,?,?)",
            (uid, name, description, color, now, now),
        )
    return get_workspace(uid)


def get_workspace(workspace_id: str) -> Optional[dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM workspaces WHERE id = ?", (workspace_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_workspace(workspace_id: str) -> bool:
    if workspace_id == DEFAULT_WORKSPACE_ID:
        return False  # Never delete default
    with _conn() as con:
        c = con.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
        return c.rowcount > 0


# ── Analyses ────────────────────────────────────────────────────────────────

def save_analysis(
    query: str,
    analysis_type: str,
    result: dict,
    name: Optional[str] = None,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
) -> dict:
    uid = _uid()
    now = _now()
    auto_name = name or _auto_name(analysis_type, query)
    with _conn() as con:
        con.execute(
            "INSERT INTO analyses VALUES (?,?,?,?,?,?,?,?,?)",
            (uid, workspace_id, analysis_type, auto_name, query,
             json.dumps(result), "complete", now, now),
        )
        # Bump workspace updated_at
        con.execute("UPDATE workspaces SET updated_at=? WHERE id=?", (now, workspace_id))
    return get_analysis(uid)


def list_analyses(workspace_id: Optional[str] = None, limit: int = 50) -> list[dict]:
    with _conn() as con:
        if workspace_id:
            rows = con.execute(
                """SELECT id, workspace_id, type, name, query, status, created_at
                   FROM analyses WHERE workspace_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (workspace_id, limit),
            ).fetchall()
        else:
            rows = con.execute(
                """SELECT id, workspace_id, type, name, query, status, created_at
                   FROM analyses ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def get_analysis(analysis_id: str) -> Optional[dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["result"] = json.loads(d.pop("result_json"))
        except Exception:
            d["result"] = {}
        return d


def delete_analysis(analysis_id: str) -> bool:
    with _conn() as con:
        c = con.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        return c.rowcount > 0


def rename_analysis(analysis_id: str, name: str) -> bool:
    with _conn() as con:
        c = con.execute(
            "UPDATE analyses SET name=?, updated_at=? WHERE id=?",
            (name, _now(), analysis_id),
        )
        return c.rowcount > 0


# ── Summary stats ───────────────────────────────────────────────────────────

def workspace_stats(workspace_id: str) -> dict:
    with _conn() as con:
        total = con.execute(
            "SELECT COUNT(*) FROM analyses WHERE workspace_id=?", (workspace_id,)
        ).fetchone()[0]
        by_type = con.execute(
            "SELECT type, COUNT(*) as n FROM analyses WHERE workspace_id=? GROUP BY type",
            (workspace_id,),
        ).fetchall()
        return {"total": total, "by_type": {r["type"]: r["n"] for r in by_type}}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _auto_name(atype: str, query: str) -> str:
    q = query[:40].strip()
    labels = {
        "variant":    f"Variant — {q}",
        "vcf":        f"Genome — {q}",
        "literature": f"Literature — {q}",
        "gene":       f"Gene — {q}",
        "chat":       f"Chat — {q}",
    }
    return labels.get(atype, f"{atype.title()} — {q}")
