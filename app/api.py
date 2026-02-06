import os
import requests
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.embeddings.vector_store import create_vector_store
from app.llm.gemini import ask_gemini
from app.db import save_chat, save_question

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = FastAPI()

# -----------------------------
# Telegram Welcome
# -----------------------------
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
            "text": "👋 Welcome to FINUX Assistant!\n\nChoose a topic or ask anything about FINUX.",
            "reply_markup": reply_markup
        }
    )

# -----------------------------
# Telegram Menus
# -----------------------------
def send_main_menu(chat_id):
    keyboard = {
        "inline_keyboard": [
            [{"text": "🚀 What is FINUX?", "callback_data": "topic:intro"}],
            [{"text": "💰 Tokenomics", "callback_data": "topic:token"}],
            [{"text": "🛠 Products", "callback_data": "topic:products"}],
            [{"text": "🧭 Roadmap", "callback_data": "topic:roadmap"}],
        ]
    }

    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": "Choose a topic 👇",
            "reply_markup": keyboard
        }
    )

def send_sub_menu(chat_id, topic):
    menus = {
        "intro": [
            [{"text": "What is FINUX?", "callback_data": "q:what is finux"}],
            [{"text": "How FINUX works?", "callback_data": "q:how finux works"}],
        ],
        "token": [
            [{"text": "Token supply", "callback_data": "q:finux token supply"}],
            [{"text": "Token utility", "callback_data": "q:finux token utility"}],
        ],
        "products": [
            [{"text": "FINUX Chatbot", "callback_data": "q:finux chatbot"}],
            [{"text": "FINUX Wallet", "callback_data": "q:finux wallet"}],
        ],
        "roadmap": [
            [{"text": "FINUX roadmap", "callback_data": "q:finux roadmap"}],
        ],
    }

    keyboard = {"inline_keyboard": menus.get(topic, [])}

    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": "Select 👇",
            "reply_markup": keyboard
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
# Load FINUX Docs (PDF + DOCX)
# -----------------------------
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

texts = []

pdf_path = os.path.join(DATA_DIR, "finux.pdf")
docx_path = os.path.join(DATA_DIR, "finux.docx")

if os.path.exists(pdf_path):
    texts += [d.page_content for d in PyPDFLoader(pdf_path).load()]

if os.path.exists(docx_path):
    texts += [d.page_content for d in Docx2txtLoader(docx_path).load()]

if not texts:
    texts = ["FINUX is a decentralized Crypto Ecosystem."]

# -----------------------------
# Vector Store
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
You are FINUX Assistant.

Answer STRICTLY using the context below.

Rules:
- Keep answers SHORT (max 4–5 lines).
- Be precise and direct.
- Use bullet points if helpful.
- Do NOT add extra explanation.
- If answer not in context, say: "Information not found."

Context:
{context}

Question:
{question}

Short Answer:
"""

    return ask_gemini(prompt)

# -----------------------------
# WEB CHAT
# -----------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.message
    answer = rag_answer(question) or ask_gemini(question)

    try:
        save_question(question)
        save_chat("web", "anonymous", "", question, answer)
    except Exception as e:
        logging.error(e)

    return {"response": answer}

# -----------------------------
@app.get("/health")
def health():
    return {"status": "FINUX chatbot running"}

# -----------------------------
# TELEGRAM WEBHOOK
# -----------------------------
@app.post("/telegram")
async def telegram_webhook(req: Request):
    data = await req.json()

    try:
        # BUTTONS
        if "callback_query" in data:
            cq = data["callback_query"]
            chat_id = cq["message"]["chat"]["id"]
            payload = cq["data"]

            if payload.startswith("topic:"):
                send_sub_menu(chat_id, payload.split(":")[1])

            elif payload.startswith("q:"):
                question = payload.split("q:")[1]
                answer = rag_answer(question) or ask_gemini(question)

                save_question(question)
                save_chat("telegram", str(chat_id), "", question, answer)

                requests.post(
                    f"{TELEGRAM_API}/sendMessage",
                    json={"chat_id": chat_id, "text": answer}
                )

            return {"ok": True}

        # NORMAL MESSAGE
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")

        if text == "/start":
            send_welcome(chat_id)
            send_main_menu(chat_id)
            return {"ok": True}

        if text:
            answer = rag_answer(text) or ask_gemini(text)

            save_question(text)
            save_chat("telegram", str(chat_id), "", text, answer)

            requests.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": chat_id, "text": answer}
            )

    except Exception as e:
        logging.error(e)

    return {"ok": True}
