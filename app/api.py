import os
import logging
import requests

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.embeddings.vector_store import create_vector_store
from app.db import save_chat, save_question

logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = FastAPI()

# ---------------- Static (logo) ----------------

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")


app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

# ---------------- CORS ----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Telegram UI ----------------

def send_welcome(chat_id):
    logo_url = "https://finux-chatbot-production.up.railway.app/static/finux.png"

    keyboard = {
        "inline_keyboard": [
            [{"text": "🚀 Open App", "url": "https://finux-chatbot-production.up.railway.app"}],
            [
                {"text": "📢 Channel", "url": "https://t.me/FINUX_ADV"},
                {"text": "🌐 Website", "url": "https://finux-chatbot-production.up.railway.app"}
            ],
        ]
    }

    requests.post(
        f"{TELEGRAM_API}/sendPhoto",
        json={
            "chat_id": chat_id,
            "photo": logo_url,
            "caption": "✨ *Welcome to FINUX*\n\nDecentralized blockchain + AI ecosystem.\n\nChoose below 👇",
            "parse_mode": "Markdown",
            "reply_markup": keyboard
        }
    )


def send_main_menu(chat_id):
    keyboard = {
        "inline_keyboard": [
            [{"text": "🚀 What is FINUX?", "callback_data": "q:what is finux"}],
            [{"text": "💰 Tokenomics", "callback_data": "q:finux tokenomics"}],
            [{"text": "🛠 Products", "callback_data": "q:finux products"}],
            [{"text": "🧭 Roadmap", "callback_data": "q:finux roadmap"}],
        ]
    }

    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": "Select 👇", "reply_markup": keyboard}
    )

# ---------------- Load FINUX Docs ----------------

pdf_path = os.path.join(RAW_DIR, "finux.pdf")
docx_path = os.path.join(RAW_DIR, "finux.docx")

documents = []

if os.path.exists(pdf_path):
    documents += PyPDFLoader(pdf_path).load()

if os.path.exists(docx_path):
    documents += Docx2txtLoader(docx_path).load()

if not documents:
    documents = [
        Document(page_content="FINUX is a decentralized crypto ecosystem.")
    ]

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)

texts = [c.page_content for c in chunks]

logging.info(f"FINUX chunks: {len(texts)}")

db = create_vector_store(texts)

# ---------------- Models ----------------

class ChatRequest(BaseModel):
    message: str

# ---------------- UI ----------------

UI_PATH = os.path.join(BASE_DIR, "ui.html")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open(UI_PATH, "r", encoding="utf-8") as f:
        return f.read()

# ---------------- RAG ----------------

def rag_answer(question: str) -> str:
    docs = db.similarity_search(question, k=3)

    if not docs:
        return ""

    text = " ".join([d.page_content for d in docs])
    text = text.replace("\n", " ")

    parts = text.split(".")[:3]

    bullets = "\n".join([f"• {p.strip()}" for p in parts if p.strip()])

    return bullets[:400]

# ---------------- Web Chat ----------------

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.message

    answer = rag_answer(question)

    if not answer:
        answer = "Not available in FINUX docs."

    try:
        save_question(question)
        save_chat("web", "anonymous", "", question, answer)
    except:
        pass

    return {"response": answer}

# ---------------- Health ----------------

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- Telegram Webhook ----------------

@app.post("/telegram")
async def telegram(req: Request):
    data = await req.json()

    try:
        if "callback_query" in data:
            cq = data["callback_query"]
            chat_id = cq["message"]["chat"]["id"]
            question = cq["data"].replace("q:", "")

        else:
            chat_id = data["message"]["chat"]["id"]
            text = data["message"].get("text", "")

            if text == "/start":
                send_welcome(chat_id)
                send_main_menu(chat_id)
                return {"ok": True}

            question = text

        answer = rag_answer(question)

        if not answer:
            answer = "Not available in FINUX docs."

        save_question(question)
        save_chat("telegram", str(chat_id), "", question, answer)

        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": answer}
        )

    except Exception as e:
        logging.error(e)

    return {"ok": True}
