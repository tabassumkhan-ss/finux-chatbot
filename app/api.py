from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import os
import re
from datetime import datetime

# FINUX loaders
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.docx_loader import load_docx
from app.ingestion.chunker import chunk_text
from app.embeddings.vector_store import create_vector_store

# Gemini
from app.llm.gemini import ask_gemini

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(title="FINUX Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL")

PDF_PATH = "data/raw/finux.pdf"
DOCX_PATH = "data/raw/finux.docx"

# -------------------------------------------------
# Build FINUX knowledge base ONCE
# -------------------------------------------------

print("Loading FINUX documents...")

pdf_text = load_pdf(PDF_PATH)
docx_text = load_docx(DOCX_PATH)

chunks = chunk_text(pdf_text + docx_text)
vector_db = create_vector_store(chunks)

print("FINUX knowledge base ready")

# -------------------------------------------------
# DB helper
# -------------------------------------------------

def save_question(question: str):
    if not DATABASE_URL:
        return

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id SERIAL PRIMARY KEY,
                question TEXT,
                created_at TIMESTAMP
            )
            """
        )

        cur.execute(
            "INSERT INTO questions (question, created_at) VALUES (%s, %s)",
            (question, datetime.utcnow()),
        )

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("DB error:", e)


# -------------------------------------------------
# Models
# -------------------------------------------------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.get("/")
def root():
    return {"status": "FINUX chatbot running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    user_message = req.message.strip()

    if not user_message:
        return {"response": "Please ask something 🙂"}

    save_question(user_message)

    try:
        # -----------------------------
        # FINUX RAG
        # -----------------------------
        docs = vector_db.similarity_search(user_message, k=3)

        if docs:
            raw = "\n\n".join(d.page_content for d in docs)
            clean = re.sub(r"\[Page\s*\d+\]", "", raw).strip()
            clean = clean[:1200]

            return {"response": clean}

        # -----------------------------
        # Gemini fallback
        # -----------------------------
        gemini_answer = ask_gemini(user_message)

        if gemini_answer and gemini_answer.strip():
            return {"response": gemini_answer}

        return {"response": "Sorry — AI service temporarily unavailable."}

    except Exception as e:
        print("Chat error:", e)
        return {"response": "Server error. Please try again."}
