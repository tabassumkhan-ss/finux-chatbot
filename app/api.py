import os
import requests
import base64
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.embeddings.vector_store import create_vector_store
from app.llm.gemini import ask_gemini
from app.db import save_chat, save_question

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = FastAPI()

def send_welcome(chat_id):
    reply_markup = {
        "inline_keyboard": [
            [
                {"text": "🚀 Open App", "url": "https://finux-chatbot-production.up.railway.app"},
                {"text": "📢 Channel", "url": "https://t.me/FINUX_ADV"}
            ],
            [
                {"text": "🌐 Website", "url": "https://finux-chatbot-production.up.railway.app"}
            ]
        ]
    }

    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": "👋 Welcome to FINUX Assistant!\n\nAsk me anything about FINUX.",
            "reply_markup": reply_markup
        }
    )


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
# Load UI
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
UI_PATH = os.path.join(BASE_DIR, "ui.html")

# -----------------------------
# Load FINUX docs
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "finux.txt")

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        texts = [f.read()]
else:
    texts = ["FINUX is a decentralized Crypto Ecosystem."]

# -----------------------------
# Vector DB
# -----------------------------
db = create_vector_store(texts)

# -----------------------------
class ChatRequest(BaseModel):
    message: str

# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    with open(UI_PATH, "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
def rag_answer(question: str) -> str:
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

# -----------------------------
# WEB CHAT
# -----------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.message

    finux_answer = rag_answer(question)

    if not finux_answer.strip():
        finux_answer = ask_gemini(question)

    answer = finux_answer or "Sorry — I could not generate a reply."

    # ALWAYS SAVE
    try:
        save_question(question)

        save_chat(
            "web",
            "anonymous",
            "",
            question,
            answer
        )
    except Exception as e:
        print("DB error:", e)

    return {"response": answer}

# -----------------------------
@app.get("/health")
def health():
    return {"status": "FINUX chatbot running"}

# -----------------------------
# TELEGRAM
# -----------------------------
@app.post("/telegram")
async def telegram_webhook(req: Request):
    data = await req.json()

    try:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")

        # 👉 Handle /start
        if text == "/start":
            send_welcome(chat_id)
            return {"ok": True}

        # Normal chat
        finux_answer = rag_answer(text)

        if not finux_answer.strip():
            finux_answer = ask_gemini(text)

        # Save Telegram chat
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
