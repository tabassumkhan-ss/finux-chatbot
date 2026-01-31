import re
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.db import init_db, save_question
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.docx_loader import load_docx
from app.ingestion.chunker import chunk_text
from app.embeddings.vector_store import create_vector_store
from app.llm.gemini import ask_gemini


app = FastAPI(title="FINUX Chatbot")

# ---------- Startup ----------

@app.on_event("startup")
def startup():
    init_db()


# ---------- UI ----------

@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/ui.html", "r", encoding="utf-8") as f:
        return f.read()


# ---------- Load FINUX docs once ----------

PDF_PATH = "data/raw/finux.pdf"
DOCX_PATH = "data/raw/finux.docx"

pdf_text = load_pdf(PDF_PATH)
docx_text = load_docx(DOCX_PATH)

chunks = chunk_text(pdf_text + docx_text)
vector_db = create_vector_store(chunks)


# ---------- Models ----------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


# ---------- Helpers ----------

def is_finux_related(q: str) -> bool:
    keywords = [
        "finux", "deposit", "referral", "club", "staking",
        "lp", "liquidity", "mint", "mst", "usdc"
    ]
    ql = q.lower()
    return any(k in ql for k in keywords)


def rag_answer(question: str) -> str | None:
    docs = vector_db.similarity_search(question, k=3)

    if not docs:
        return None

    text = "\n".join(d.page_content for d in docs)
    text = re.sub(r"\[Page\s*\d+\]", "", text)
    text = text.strip()

    return text[:1500]


# ---------- Chat ----------

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.message.strip()

    if not question:
        return {"response": "Please ask something 🙂"}

    # 1️⃣ Try FINUX RAG first
    finux_answer = rag_answer(question)

    if finux_answer and finux_answer.strip():
        answer = finux_answer

    else:
        # 2️⃣ Fallback to Gemini
        gemini_answer = ask_gemini(question)

        if gemini_answer and gemini_answer.strip():
            answer = gemini_answer
        else:
            answer = "Sorry — I could not generate a reply."

    return {"response": answer}
