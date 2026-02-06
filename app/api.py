import os
import requests
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
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

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Telegram UI ----------------

def send_welcome(chat_id):
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": "👋 Welcome to FINUX Assistant!\n\nChoose a topic or ask about FINUX."
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

# ---------------- Load UI ----------------

BASE_DIR = os.path.dirname(__file__)
UI_PATH = os.path.join(BASE_DIR, "ui.html")

# ---------------- Load FINUX Docs ----------------

DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
pdf_path = os.path.join(DATA_DIR, "finux.pdf")
docx_path = os.path.join(DATA_DIR, "finux.docx")

documents = []

if os.path.exists(pdf_path):
    documents += PyPDFLoader(pdf_path).load()

if os.path.exists(docx_path):
    documents += Docx2txtLoader(docx_path).load()

if not documents:
    documents = [
        Document(page_content="FINUX is a decentralized crypto ecosystem providing blockchain products, token utilities, and AI tools.")
    ]

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)

texts = [c.page_content for c in chunks]
logging.info(f"Loaded FINUX chunks: {len(texts)}")

# ---------------- Vector Store ----------------

db = create_vector_store(texts)

# ---------------- Models ----------------

class ChatRequest(BaseModel):
    message: str

# ---------------- Pages ----------------

@app.get("/", response_class=HTMLResponse)
async def home():
    with open(UI_PATH, "r", encoding="utf-8") as f:
        return f.read()

# ---------------- RAG ----------------

def rag_answer(question: str) -> str:
    docs = db.similarity_search(question, k=3)

    if not docs:
        return "Not available in FINUX docs."

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer ONLY using this FINUX context.

Rules:
- Max 3 bullet points
- Very short
- No outside knowledge
- If answer not clearly present, reply exactly:
Not available in FINUX docs.

Context:
{context}

Question:
{question}

Answer:
"""

    return ask_gemini(prompt)

# ---------------- Web Chat ----------------

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.message

    answer = rag_answer(question)

    if not answer:
        answer = ask_gemini(question)

    if not answer:
        answer = "Sorry — I couldn’t find an answer about FINUX."

    try:
        save_question(question)
        save_chat("web", "anonymous", "", question, answer)
    except Exception as e:
        logging.error(e)

    return {"response": answer}

# ---------------- Health ----------------

@app.get("/health")
def health():
    return {"status": "FINUX chatbot running"}

# ---------------- Telegram Webhook ----------------

@app.post("/telegram")
async def telegram_webhook(req: Request):
    data = await req.json()

    try:
        # Button click
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
            answer = ask_gemini(question)

        if not answer:
            answer = "Sorry — I couldn’t find an answer about FINUX."

        save_question(question)
        save_chat("telegram", str(chat_id), "", question, answer)

        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": answer}
        )

    except Exception as e:
        logging.error(e)

    return {"ok": True}
