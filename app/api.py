from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import os
from datetime import datetime

# Your helpers
from app.llm.gemini import ask_gemini
from app.rag import get_rag_answer

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(title="Finux Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL")

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

    # Save question
    save_question(user_message)

    try:
        # -----------------------------
        # 1. FINUX RAG first
        # -----------------------------
        rag_answer = get_rag_answer(user_message)

        if rag_answer and rag_answer.strip():
            return {"response": rag_answer}

        # -----------------------------
        # 2. Gemini fallback
        # -----------------------------
        gemini_answer = ask_gemini(user_message)

        if gemini_answer and gemini_answer.strip():
            return {"response": gemini_answer}

        # -----------------------------
        # 3. Final fallback
        # -----------------------------
        return {
            "response": "Sorry — AI service is temporarily unavailable. Please try again."
        }

    except Exception as e:
        print("Chat error:", e)

        # NEVER return None
        return {
            "response": "Something went wrong on server. Please try again."
        }
