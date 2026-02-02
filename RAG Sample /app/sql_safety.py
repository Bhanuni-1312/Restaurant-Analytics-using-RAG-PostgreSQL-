from __future__ import annotations

import re


class SQLSafetyError(ValueError):
    pass


_FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "VACUUM", "ANALYZE",
    "COPY", "CALL", "EXEC", "EXECUTE",
]

_FORBIDDEN_SCHEMAS = [
    "pg_catalog",
    "information_schema",
]


def validate_sql(sql: str) -> str:
    """
    Hard guardrails before execution.
    - Only SELECT or WITH ... SELECT
    - No multiple statements
    - Block dangerous keywords
    - Block system schemas
    """
    if sql is None:
        raise SQLSafetyError("SQL is empty.")

    s = sql.strip()
    if not s:
        raise SQLSafetyError("SQL is empty.")

    # No multiple statements: disallow semicolon not at end or multiple semicolons
    # Allow at most one trailing semicolon.
    semicolons = s.count(";")
    if semicolons > 1:
        raise SQLSafetyError("Multiple SQL statements are not allowed.")
    if semicolons == 1 and not s.endswith(";"):
        raise SQLSafetyError("Semicolon must be only at the end (single statement).")

    # Remove trailing semicolon for easier checks
    s_no_sc = s[:-1].strip() if s.endswith(";") else s

    # Must start with SELECT or WITH
    head = s_no_sc[:20].lstrip().upper()
    if not (head.startswith("SELECT") or head.startswith("WITH")):
        raise SQLSafetyError("Only SELECT queries are allowed.")

    upper = s_no_sc.upper()

    # Block forbidden keywords (word-boundary)
    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", upper):
            raise SQLSafetyError(f"Forbidden keyword detected: {kw}")

    # Block system schemas
    for schema in _FORBIDDEN_SCHEMAS:
        if re.search(rf"\b{re.escape(schema.upper())}\b", upper):
            raise SQLSafetyError(f"Access to forbidden schema detected: {schema}")

    return s_no_sc
