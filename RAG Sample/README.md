RAG over PostgreSQL using ChromaDB + local embeddings, Text-to-SQL via OpenAI, served as an API.

Steps:

1) Create DB & tables in PostgreSQL (pgAdmin or CLI)
2) Copy .env.example → .env and fill values (OpenAI + Postgres)
3) Install dependencies:
   pip install -r requirements.txt

4) Run API:
   uvicorn app.api:app --reload

5) Ask question:
   POST http://127.0.0.1:8000/ask
   Body JSON:
   { "question": "..." }

Optional:
- GET http://127.0.0.1:8000/health