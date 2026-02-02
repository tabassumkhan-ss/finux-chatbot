import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.embeddings.vector_store import create_vector_store
from app.llm.gemini import ask_gemini
from app.db import save_chat

# -----------------------------
# Telegram
# -----------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = FastAPI()

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# UI path
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
UI_PATH = os.path.join(BASE_DIR, "ui.html")

# -----------------------------
# Load FINUX data
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "finux.txt")

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        texts = [f.read()]
else:
    texts = ["FINUX is a decentralized crypto ecosystem."]

# -----------------------------
# Build Vector Store ONCE
# -----------------------------
print("Building vector store...")
db = create_vector_store(texts)
print("Vector store ready.")

# -----------------------------
# Request schema
# -----------------------------
class ChatRequest(BaseModel):
    message: str

# -----------------------------
# Home → UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    with open(UI_PATH, "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
# RAG
# -----------------------------
def rag_answer(question: str) -> str:
    try:
        docs = db.similarity_search_with_score(question, k=2)

        if not docs:
            return ""

        context = "\n".join([d[0].page_content for d in docs])

        prompt = f"""
Use the context below to answer.

Context:
{context}

Question:
{question}
"""

        return ask_gemini(prompt)

    except Exception as e:
        print("RAG error:", e)
        return ""

# -----------------------------
# Web Chat
# -----------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.message

    # RAG first
    answer = rag_answer(question)

    # Fallback Gemini
    if not answer.strip():
        answer = ask_gemini(question)

    if not answer.strip():
        answer = "Sorry — I could not generate a reply."

    # Save to DB (WEB)
    try:
        save_chat(
            "web",
            "anonymous",
            "",
            question,
            answer
        )
    except Exception as e:
        print("DB save error (web):", e)

    return {"response": answer}

# -----------------------------
# Telegram Webhook
# -----------------------------
@app.post("/telegram")
async def telegram_webhook(req: Request):
    data = await req.json()

    try:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"]["text"]

        answer = rag_answer(text)

        if not answer.strip():
            answer = ask_gemini(text)

        if not answer.strip():
            answer = "Sorry — I could not generate a reply."

        # Send to Telegram
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": answer
            }
        )

        # Save to DB (Telegram)
        try:
            save_chat(
                "telegram",
                str(chat_id),
                "",
                text,
                answer
            )
        except Exception as e:
            print("DB save error (telegram):", e)

    except Exception as e:
        print("Telegram error:", e)

    return {"ok": True}

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "FINUX chatbot running"}
