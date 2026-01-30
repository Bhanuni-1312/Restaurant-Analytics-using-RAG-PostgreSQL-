import json
import os
import re
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.db import run_sql
from app.openai_client import get_client, get_chat_model

# --- Settings ---
SHOW_SQL = os.getenv("SHOW_SQL", "false").lower() in ("1", "true", "yes", "y")
CHROMA_DIR = os.getenv("CHROMA_DIR", "storage/chroma")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "schema_docs")
SCHEMA_DOCS_PATH = os.getenv("SCHEMA_DOCS_PATH", "data/schema_docs.json")

# Silence tokenizer fork warning (harmless but noisy)
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")


def clean_sql(sql_text: str) -> str:
    """
    Removes markdown fences and stray backticks.
    """
    sql_text = (sql_text or "").strip()

    # Remove fenced blocks like ```sql ... ```
    if sql_text.startswith("```"):
        sql_text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", sql_text)
        sql_text = re.sub(r"\s*```$", "", sql_text)

    # Remove wrapping backticks if any
    sql_text = sql_text.strip("` \n\t")
    print("sql text :\n" + sql_text)
    return sql_text


def load_schema_docs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Expecting: {"tables":[{...},{...}]}
    tables = data.get("tables", [])
    if not isinstance(tables, list) or not tables:
        raise ValueError("schema_docs.json must contain a non-empty 'tables' list.")
    return tables


def schema_doc_to_text(table_doc: Dict[str, Any]) -> str:
    """
    Converts a table doc into a text chunk used for retrieval.
    We include both table + column descriptions for best grounding.
    """
    table_name = table_doc.get("table_name", "")
    table_desc = table_doc.get("description", "")

    cols = table_doc.get("columns", [])
    col_lines = []
    for c in cols:
        col_lines.append(
            f"- {c.get('name')} ({c.get('type')}): {c.get('description','')}"
        )

    return (
        f"TABLE: {table_name}\n"
        f"DESCRIPTION: {table_desc}\n"
        f"COLUMNS:\n" + "\n".join(col_lines)
    )


class SchemaRAG:
    def __init__(self):
        # local embeddings (no OpenAI embeddings required)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # chroma local persistent store
        self.chroma = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(COLLECTION_NAME)

        self.oai = get_client()
        self.chat_model = get_chat_model()

    def index_schema(self):
        """
        Index schema_docs.json into Chroma. Safe to run multiple times.
        We upsert by deterministic ids (table name).
        """
        tables = load_schema_docs(SCHEMA_DOCS_PATH)

        ids = []
        docs = []
        embeds = []

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

        # Chroma add() fails if ids exist; use upsert()
        self.collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=embeds,
        )

    def retrieve_schema(self, question: str, top_k: int = 2) -> List[str]:
        q_emb = self.embedder.encode(question).tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        return docs

    def generate_sql(self, question: str, schema_context: List[str]) -> str:
        """
        Uses chat model to generate SQL. We force raw SQL output (no markdown).
        """
        system = (
            "You are a senior data engineer. "
            "You write correct PostgreSQL SQL queries using ONLY the provided schema context. "
            "Safety rules:\n"
            "- Output ONLY raw SQL text. No markdown fences, no ```sql, no explanations.\n"
            "- Only SELECT (or WITH...SELECT). Never INSERT/UPDATE/DELETE/DDL.\n"
            "- Use correct table/column names exactly as in the schema.\n"
        )

        context = "\n\n".join(schema_context)

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
        sql = resp.choices[0].message.content or ""
        return clean_sql(sql)

    def summarize_results(self, question: str, sql: str, rows: List[Dict[str, Any]]) -> str:
        """
        Uses chat model to convert SQL result rows into a human answer.
        """
        system = (
            "You are a helpful analytics assistant. "
            "Given a user's question and SQL result rows, answer clearly and concisely. "
            "Do not mention SQL unless the user explicitly asks."
        )

        user = (
            f"QUESTION:\n{question}\n\n"
            f"ROW_COUNT: {len(rows)}\n"
            f"ROWS (JSON):\n{json.dumps(rows, ensure_ascii=False, indent=2)}\n\n"
            "Answer in plain English."
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

    def answer_question(self, question: str) -> str:
        schema_hits = self.retrieve_schema(question, top_k=2)
        sql = self.generate_sql(question, schema_hits)

        if SHOW_SQL:
            print("\n[DEBUG] Generated SQL:\n", sql)

        rows = run_sql(sql)
        answer = self.summarize_results(question, sql, rows)
        return answer
        