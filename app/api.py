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

app = FastAPI(title="FINUX Chatbot API")


# ---------------- Startup ----------------

@app.on_event("startup")
def on_startup():
    init_db()


# ---------------- UI ----------------

@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/ui.html", "r", encoding="utf-8") as f:
        return f.read()


# ---------------- Knowledge Base ----------------

PDF_PATH = "data/raw/finux.pdf"
DOCX_PATH = "data/raw/finux.docx"

pdf_text = load_pdf(PDF_PATH)
docx_text = load_docx(DOCX_PATH)
chunks = chunk_text(pdf_text + docx_text)
db = create_vector_store(chunks)


# ---------------- Models ----------------

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# ---------------- Chat Endpoint ----------------

@app.post("/chat")
def chat(req: ChatRequest):
    save_question(req.question)

    question = req.question.strip()
    q = question.lower()

    # Search FINUX docs
    docs = db.similarity_search(question, k=3)

    finux_context = ""
    if docs:
        raw = "\n\n".join(d.page_content for d in docs)
        clean = re.sub(r"\[Page\s*\d+\]", "", raw)
        clean = "\n".join(dict.fromkeys(clean.splitlines()))
        finux_context = clean[:1500]

    # Detect language (simple)
    hinglish = ["kya","ka","ke","me","mujhe","samjha","bata","hai","hain","kar","kaise"]
    is_hinglish = any(w in q for w in hinglish)
    is_english = not is_hinglish

    # Decide mode
    is_finux = len(finux_context.strip()) > 150

    if is_finux:
        system = (
            "You are FINUX Assistant. Answer ONLY using provided FINUX content. "
            "Be friendly, simple, human. No marketing. No repetition."
        )

        if not is_english:
            system += " Respond in Hinglish."

        prompt = f"""
{system}

FINUX CONTENT:
{finux_context}

USER QUESTION:
{question}

Answer clearly:
"""

    else:
        prompt = question

    answer = ask_gemini(prompt)

    return {"answer": answer}
