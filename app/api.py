import os
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import psycopg2

from app.llm.gemini import ask_gemini
from app.embeddings.vector_store import get_rag_answer
from fastapi.responses import HTMLResponse

DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI()


# ---------------- Models ----------------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# ---------------- Database ----------------

def get_db():
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL)


def save_question(question: str):
    try:
        conn = get_db()
        if not conn:
            return

        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id SERIAL PRIMARY KEY,
                question TEXT,
                created_at TIMESTAMP
            )
        """)

        cur.execute(
            "INSERT INTO questions (question, created_at) VALUES (%s, %s)",
            (question, datetime.utcnow()),
        )

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("DB error:", e)


# ---------------- Routes ----------------

@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/ui.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    user_msg = req.message.strip()

    if not user_msg:
        return {"response": "Please enter a message."}

    # save question
    save_question(user_msg)

    finux_answer = ""
    gemini_answer = ""

    # 1. FINUX RAG
    try:
        finux_answer = get_rag_answer(user_msg)
    except Exception as e:
        print("RAG error:", e)

    if finux_answer and finux_answer.strip():
        return {"response": finux_answer}

    # 2. Gemini fallback
    try:
        gemini_answer = ask_gemini(user_msg)
    except Exception as e:
        print("Gemini error:", e)

    if gemini_answer and gemini_answer.strip():
        return {"response": gemini_answer}

    # 3. Final fallback
    return {"response": "Sorry — AI service temporarily unavailable."}
