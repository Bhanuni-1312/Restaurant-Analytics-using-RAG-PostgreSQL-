import json
import os
import re
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.db import run_sql
from app.openai_client import get_client, get_chat_model


SHOW_SQL = os.getenv("SHOW_SQL", "false").lower() in ("1", "true", "yes", "y")
CHROMA_DIR = os.getenv("CHROMA_DIR", "storage/chroma")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "schema_docs")
SCHEMA_DOCS_PATH = os.getenv("SCHEMA_DOCS_PATH", "data/schema_docs.json")

os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")


# -----------------------------
# Helpers
# -----------------------------
def clean_sql(sql_text: str) -> str:
    sql_text = (sql_text or "").strip()

    # remove ```sql fences
    if sql_text.startswith("```"):
        sql_text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", sql_text)
        sql_text = re.sub(r"\s*```$", "", sql_text)

    # remove stray backticks and extra whitespace
    sql_text = sql_text.strip("` \n\t")
    return sql_text


def load_schema_docs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tables = data.get("tables", [])
    if not isinstance(tables, list) or not tables:
        raise ValueError("schema_docs.json must contain a non-empty 'tables' list.")
    return tables


def schema_doc_to_text(table_doc: Dict[str, Any]) -> str:
    table_name = table_doc.get("table_name", "")
    table_desc = table_doc.get("description", "")
    cols = table_doc.get("columns", [])
    col_lines = []
    for c in cols:
        col_lines.append(f"- {c.get('name')} ({c.get('type')}): {c.get('description','')}")
    return (
        f"TABLE: {table_name}\n"
        f"DESCRIPTION: {table_desc}\n"
        f"COLUMNS:\n" + "\n".join(col_lines)
    )


# ---- SQL safety guardrails ----
DISALLOWED_SQL = (
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "commit", "rollback"
)

def validate_sql(sql: str) -> None:
    """
    Safety validation before executing model output.
    - Single statement only
    - Must be SELECT / WITH
    - Must not contain dangerous keywords
    """
    s = (sql or "").strip()
    if not s:
        raise ValueError("Empty SQL generated.")

    # single statement (basic heuristic)
    if ";" in s[:-1]:
        raise ValueError("Rejected: multiple SQL statements detected.")

    low = s.lower().strip()
    if not (low.startswith("select") or low.startswith("with")):
        raise ValueError("Rejected: only SELECT/WITH queries are allowed.")

    for kw in DISALLOWED_SQL:
        if re.search(rf"\b{kw}\b", low):
            raise ValueError(f"Rejected: disallowed keyword in SQL: {kw}")


def should_include_aggregate(question: str) -> bool:
    """
    If question implies ranking/top/highest/most/count/average etc.,
    we should include evidence columns (count, sum, avg) in SELECT.
    """
    q = (question or "").lower()
    triggers = [
        "highest", "lowest", "most", "least", "top", "maximum", "minimum",
        "count", "how many", "total", "average", "avg", "sum", "rank"
    ]
    return any(t in q for t in triggers)


class SchemaRAG:
    def __init__(self):
        # Lazy init flags (embedding model load is expensive)
        self._embedder = None

        self.chroma = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(COLLECTION_NAME)

        self.oai = get_client()
        self.chat_model = get_chat_model()

    @property
    def embedder(self) -> SentenceTransformer:
        # Lazy load only when actually needed
        if self._embedder is None:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def index_schema(self) -> None:
        tables = load_schema_docs(SCHEMA_DOCS_PATH)

        ids: List[str] = []
        docs: List[str] = []
        embeds: List[List[float]] = []

        for t in tables:
            table_name = t.get("table_name")
            if not table_name:
                continue
            ids.append(f"table::{table_name}")
            txt = schema_doc_to_text(t)
            docs.append(txt)
            embeds.append(self.embedder.encode(txt).tolist())

        if not ids:
            raise ValueError("No valid tables found in schema_docs.json to index.")

        self.collection.upsert(ids=ids, documents=docs, embeddings=embeds)

    def retrieve_schema(self, question: str, top_k: int = 2) -> List[str]:
        q_emb = self.embedder.encode(question).tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        return res.get("documents", [[]])[0]

    def generate_sql(self, question: str, schema_context: List[str]) -> str:
        """
        Improve prompt so SQL includes evidence when question implies aggregates/ranking.
        """
        context = "\n\n".join(schema_context)

        evidence_rule = ""
        if should_include_aggregate(question):
            evidence_rule = (
                "\n- IMPORTANT: If the question asks for highest/most/top/count/total/average etc., "
                "the SELECT must include the metric column used for ranking (e.g., COUNT(*) AS cnt, SUM(x) AS total). "
                "Do not return only the label column; return both label and metric.\n"
            )

        system = (
            "You are a senior data engineer. "
            "You write correct PostgreSQL SELECT queries using ONLY the provided schema context.\n"
            "Rules:\n"
            "- Output ONLY raw SQL text. No markdown, no explanation.\n"
            "- Only SELECT (or WITH...SELECT). No DDL/DML.\n"
            "- Use exact table/column names from schema.\n"
            "- Add LIMIT when returning ranked lists.\n"
            f"{evidence_rule}"
        )

        user = (
            f"SCHEMA CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Return ONLY the SQL query."
        )

        resp = self.oai.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )

        # normalize response extraction
        sql = (resp.choices[0].message.content or "").strip()
        sql = clean_sql(sql)

        validate_sql(sql)
        return sql

    def summarize_results(self, question: str, sql: str, rows: List[Dict[str, Any]]) -> str:
        """
        Force the LLM to ground the answer in the provided row fields.
        """
        system = (
            "You are a helpful analytics assistant. "
            "Answer the user's question using ONLY the provided result rows. "
            "If a numeric metric exists in the rows (count/total/avg/sum/etc.), you MUST mention it in the answer. "
            "Do not invent numbers. Do not mention SQL unless user asks."
        )

        user = (
            f"QUESTION:\n{question}\n\n"
            f"ROWS (JSON):\n{json.dumps(rows, ensure_ascii=False, indent=2)}\n\n"
            "Write a single concise sentence answering the question. "
            "If rows contain both a label (e.g., city) and metric (e.g., order_count), include both."
        )

        resp = self.oai.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()

    def answer_question(self, question: str) -> Dict[str, Any]:
        # Retrieve schema
        schema_hits = self.retrieve_schema(question, top_k=2)

        # Generate SQL (with evidence rules + validation)
        sql = self.generate_sql(question, schema_hits)

        if SHOW_SQL:
            print("\n[DEBUG] Generated SQL:\n", sql)

        # Execute
        rows = run_sql(sql)

        # Summarize grounded in rows (numbers included if present)
        answer = self.summarize_results(question, sql, rows)

        # ✅ Return clean structure for API consumers
        return {"answer": answer, "sql": sql, "rows": rows}