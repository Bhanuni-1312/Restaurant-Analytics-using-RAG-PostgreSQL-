import json
import os
import re
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.db import run_sql
from app.openai_client import get_client, get_chat_model
from app.sql_safety import validate_sql  # ✅ single source of truth for SQL safety


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
        # Lazy init flags (you don't want to load them unless you actually need)
        self._embedder = None

        # ✅ NEW: track whether we ensured schema is indexed for this process
        self._schema_indexed = False

        self.chroma = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(COLLECTION_NAME)

        self.oai = get_client()  # creating an authenticated OpenAI client
        self.chat_model = get_chat_model()  # model name is got from .env

    @property
    def embedder(self) -> SentenceTransformer:
        # Lazy load only when actually needed
        if self._embedder is None:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    # ensure schema is indexed at least once
    def ensure_schema_indexed(self) -> None:
        if self._schema_indexed:
            return

        try:
            existing = self.collection.count()
        except Exception:
            existing = 0

        # If collection is empty, index schema_docs.json
        if existing == 0:
            self.index_schema()

        self._schema_indexed = True

    def index_schema(self) -> None:
        # converts schema -> embeddings -> and store it in Chroma
        tables = load_schema_docs(SCHEMA_DOCS_PATH)  # knowledge base

        ids: List[str] = []             # unique keys
        docs: List[str] = []            # textual schema representation
        embeds: List[List[float]] = []  # vectors

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

        sql = (resp.choices[0].message.content or "").strip()
        sql = clean_sql(sql)

        # ✅ CHANGED: use ONLY sql_safety.validate_sql as the single validator
        # Depending on your sql_safety.py, it may return cleaned SQL. We handle both safely.
        validated = validate_sql(sql)
        if isinstance(validated, str) and validated.strip():
            sql = validated.strip()

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
        # ✅ NEW: make sure schema is in Chroma before retrieval
        self.ensure_schema_indexed()

        schema_hits = self.retrieve_schema(question, top_k=2)
        if not schema_hits:
            raise ValueError("No schema context retrieved. Check schema_docs.json indexing/Chroma path.")

        sql = self.generate_sql(question, schema_hits)

        if SHOW_SQL:
            print("\n[DEBUG] Generated SQL:\n", sql)

        rows = run_sql(sql)
        answer = self.summarize_results(question, sql, rows)

        return {"answer": answer, "sql": sql, "rows": rows}