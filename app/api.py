import os
import re
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.db import init_db, save_question

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.docx_loader import load_docx
from app.ingestion.chunker import chunk_text
from app.embeddings.vector_store import create_vector_store

from app.llm.gemini import ask_gemini

# ---------------- App ----------------

app = FastAPI(title="FINUX Chatbot API")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/ui.html", "r", encoding="utf-8") as f:
        return f.read()


# ---------------- Load FINUX Docs Once ----------------

PDF_PATH = "data/raw/finux.pdf"
DOCX_PATH = "data/raw/finux.docx"

pdf_text = load_pdf(PDF_PATH)
docx_text = load_docx(DOCX_PATH)

chunks = chunk_text(pdf_text + docx_text)
vector_db = create_vector_store(chunks)


# ---------------- Models ----------------

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# ---------------- Helpers ----------------

def detect_language(text: str):
    q = text.lower()
    hinglish = ["kya", "ka", "ke", "me", "mujhe", "kaise", "bata", "hai", "hain", "kar"]
    if any(w in q for w in hinglish):
        return "hinglish"
    if re.search(r"[a-zA-Z]", text):
        return "english"
    return "hinglish"


def get_finux_answer(question: str) -> str:
    docs = vector_db.similarity_search(question, k=3)

    if not docs:
        return ""

    raw = "\n\n".join(dict.fromkeys(d.page_content for d in docs))
    clean = re.sub(r"\[Page\s*\d+\]", "", raw)
    clean = "\n".join(dict.fromkeys(clean.splitlines()))
    return clean.strip()[:1200]


# ---------------- Chat Endpoint ----------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    user_q = req.question.strip()
    save_question(user_q)

    lang = detect_language(user_q)
    q = user_q.lower()

    finux_keywords = [
        "finux", "deposit", "referral", "staking", "lp", "club",
        "reward", "mst", "usdc", "wallet", "mining"
    ]

    is_finux = any(k in q for k in finux_keywords)

    if is_finux:
        final_core = get_finux_answer(user_q)
        if not final_core:
            final_core = "Sorry — FINUX documents me iska jawab nahi mila."
    else:
        final_core = ask_gemini(user_q)

    if not final_core:
        return {"answer": "Sorry — service temporarily unavailable."}
