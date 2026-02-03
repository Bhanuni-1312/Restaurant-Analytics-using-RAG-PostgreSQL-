from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware# allows frontend browser to call your API 
from pydantic import BaseModel, Field

from app.rag import SchemaRAG

app = FastAPI(title="SchemaRAG API")

# Dev-friendly CORS(cross origin resourse sharing- access securly resources from different domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#api == controller ; rag == engine (object is created here)
rag = SchemaRAG()

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")

class AskResponse(BaseModel):
    answer: str
    sql: str
    rows: list

@app.get("/health")
#just to check if the server is running or not
def health():
    return {"ok": True, "message": "API running"}

@app.post("/ask", response_model=AskResponse)
#the question is converted into req.question by FastAPI 
def ask(req: AskRequest):
    #cleans the text(here question)
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        result = rag.answer_question(q)
        # result is expected to be dict: {"answer":..., "sql":..., "rows":...}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))