from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag import SchemaRAG

app = FastAPI(title="SchemaRAG API")

# Dev-friendly CORS (restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = SchemaRAG()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")


class AskResponse(BaseModel):
    answer: str
    sql: str
    rows: list


@app.get("/health")
def health():
    return {"ok": True, "message": "API running"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        result = rag.answer_question(q)
        # result is expected to be dict: {"answer":..., "sql":..., "rows":...}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))