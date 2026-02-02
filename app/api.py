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

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
UI_PATH = os.path.join(BASE_DIR, "ui.html")
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "finux.txt")

# -----------------------------
# Load FINUX Text
# -----------------------------
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        texts = [f.read()]
else:
    texts = ["FINUX is a decentralized crypto ecosystem."]

# -----------------------------
# Build Vector Store ONCE
# -----------------------------
db = create_vector_store(texts)

# -----------------------------
# Request Model
# -----------------------------
class ChatRequest(BaseModel):
    message: str

# -----------------------------
# UI
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

    finux_answer = rag_answer(question)

    if finux_answer and finux_answer.strip():
        answer = finux_answer
    else:
        gemini_answer = ask_gemini(question)
        if gemini_answer and gemini_answer.strip():
            answer = gemini_answer
        else:
            answer = "Sorry — I could not generate a reply."

    # ✅ Save WEB chat
    save_chat(
        "web",
        "anonymous",
        "",
        question,
        answer
    )

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

        finux_answer = rag_answer(text)

        if not finux_answer.strip():
            finux_answer = ask_gemini(text)

        # ✅ Save TELEGRAM chat
        save_chat(
            "telegram",
            str(chat_id),
            "",
            text,
            finux_answer
        )

        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": finux_answer
            }
        )

    except Exception as e:
        print("Telegram error:", e)

    return {"ok": True}

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "FINUX chatbot running"}
