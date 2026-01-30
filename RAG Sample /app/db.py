# app/db.py
from __future__ import annotations

import os
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def get_conn():
    """
    Create and return a new PostgreSQL connection using env vars.

    Use either:
      - DATABASE_URL  (preferred)
    OR:
      - PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
    """
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url)

    host = _get_env("PGHOST", "localhost")
    port = int(_get_env("PGPORT", "5432"))
    dbname = _get_env("PGDATABASE")
    user = _get_env("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )


def _json_safe(value: Any) -> Any:
    """Convert DB-returned types into JSON-serializable Python types."""
    if isinstance(value, Decimal):
        return float(value)  # or str(value) if you want exact precision
    return value


def _row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _json_safe(v) for k, v in row.items()}


def fetch_all(sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
    """Run a SELECT and return list of dict rows (JSON-safe)."""
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
            return [_row_to_dict(dict(r)) for r in rows]
    finally:
        conn.close()


def fetch_one(sql: str, params: Optional[Sequence[Any]] = None) -> Optional[Dict[str, Any]]:
    """Run a SELECT and return one dict row (JSON-safe) or None."""
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or ())
            row = cur.fetchone()
            return _row_to_dict(dict(row)) if row else None
    finally:
        conn.close()


def execute(sql: str, params: Optional[Sequence[Any]] = None) -> int:
    """Run INSERT/UPDATE/DELETE and return rowcount."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            conn.commit()
            return cur.rowcount
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# -----------------------------
# Backward-compatible API
# -----------------------------
def run_sql(
    sql: str,
    params: Optional[Sequence[Any]] = None,
    *,
    fetch: str = "all",
) -> Union[List[Dict[str, Any]], Optional[Dict[str, Any]], int]:
    """
    Compatibility wrapper for older code expecting `run_sql`.

    fetch:
      - "all"  -> list of rows
      - "one"  -> single row or None
      - "none" -> execute (returns rowcount)
    """
    fetch = (fetch or "all").lower().strip()
    if fetch == "all":
        return fetch_all(sql, params)
    if fetch == "one":
        return fetch_one(sql, params)
    if fetch in ("none", "execute"):
        return execute(sql, params)
    raise ValueError("fetch must be one of: 'all', 'one', 'none'")


def healthcheck() -> Tuple[bool, str]:
    """Simple connectivity check."""
    try:
        row = fetch_one("SELECT 1 AS ok;")
        if row and row.get("ok") == 1:
            return True, "DB connection OK"
        return False, "DB connection failed (unexpected response)"
    except Exception as e:
        return False, f"DB connection failed: {e}"