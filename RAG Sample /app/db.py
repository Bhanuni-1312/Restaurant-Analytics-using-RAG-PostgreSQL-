# app/db.py
from __future__ import annotations

import os
import threading
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Pool config
# -----------------------------
_POOL: Optional[SimpleConnectionPool] = None
_POOL_LOCK = threading.Lock()

DEFAULT_MINCONN = int(os.getenv("PGPOOL_MINCONN", "1"))
DEFAULT_MAXCONN = int(os.getenv("PGPOOL_MAXCONN", "5"))

# Choose how Decimals are represented in JSON:
# - "string" keeps exact precision (recommended for money/ratings)
# - "float" makes it numeric but can lose precision slightly
DECIMAL_JSON_MODE = (os.getenv("DECIMAL_JSON_MODE") or "string").strip().lower()


def _get_env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def _make_dsn() -> str:
    """
    Builds a DSN string for psycopg2.
    Priority:
      1) DATABASE_URL
      2) PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
    """
    database_url = (os.getenv("DATABASE_URL") or "").strip()
    if database_url:
        return database_url

    host = _get_env("PGHOST", "localhost")
    port = _get_env("PGPORT", "5432")
    dbname = _get_env("PGDATABASE")
    user = _get_env("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "")

    # DSN format
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def get_pool() -> SimpleConnectionPool:
    """
    Creates (if needed) and returns a global connection pool.
    Thread-safe initialization.
    """
    global _POOL
    if _POOL is not None:
        return _POOL

    with _POOL_LOCK:
        if _POOL is None:
            dsn = _make_dsn()
            _POOL = SimpleConnectionPool(
                minconn=DEFAULT_MINCONN,
                maxconn=DEFAULT_MAXCONN,
                dsn=dsn,
            )
    return _POOL


def get_conn():
    """
    Gets a connection from the pool.
    Must be returned via put_conn().
    """
    pool = get_pool()
    return pool.getconn()


def put_conn(conn) -> None:
    """
    Returns a connection back to the pool.
    """
    pool = get_pool()
    pool.putconn(conn)


def close_pool() -> None:
    """
    Close all pool connections (rarely needed in dev, useful in shutdown hooks).
    """
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.closeall()
            _POOL = None


# -----------------------------
# JSON-safe conversion
# -----------------------------
def _json_safe(value: Any) -> Any:
    """
    Convert DB types into JSON-serializable types.
    """
    if isinstance(value, Decimal):
        if DECIMAL_JSON_MODE == "float":
            return float(value)
        return str(value)  # exact
    return value


def _row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _json_safe(v) for k, v in row.items()}


# -----------------------------
# Query helpers
# -----------------------------
def fetch_all(sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
    """
    Run a SELECT and return list of dict rows (JSON-safe).
    Uses pooled connection.
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
            return [_row_to_dict(dict(r)) for r in rows]
    finally:
        put_conn(conn)


def fetch_one(sql: str, params: Optional[Sequence[Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Run a SELECT and return one dict row (JSON-safe) or None.
    Uses pooled connection.
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or ())
            row = cur.fetchone()
            return _row_to_dict(dict(row)) if row else None
    finally:
        put_conn(conn)


def execute(sql: str, params: Optional[Sequence[Any]] = None) -> int:
    """
    Run INSERT/UPDATE/DELETE and return rowcount.
    Uses pooled connection.
    """
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
        put_conn(conn)


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
    """
    Simple DB connectivity check.
    """
    try:
        row = fetch_one("SELECT 1 AS ok;")
        if row and row.get("ok") == 1:
            return True, "DB connection OK"
        return False, "DB connection failed (unexpected response)"
    except Exception as e:
        return False, f"DB connection failed: {e}"