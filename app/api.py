import os
import logging
import requests
import httpx

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

MENUS = {
    "main": {
        "Deposit": "menu:deposit",
        "Fund Distribution": "menu:funds",
        "Rank Wise Rewards": "menu:ranks",
        "Dual Income System": "menu:dual_income",
        "Others": "menu:others"
    },

    "deposit": {
        "Deposit Slab": "q:deposit_slab",
        "Self-Staking": "q:self_staking",
        "Min. Deposit & Structure": "q:min_deposit",
        "⬅️ Back": "menu:main"
    },

    "funds": {
        "MSTC": "q:mstc",
        "USDCe": "q:usdce",
        "Fund Distribution": "q:fund_distribution",
        "⬅️ Back": "menu:main"
    },

    "ranks": {
        "Origin": "q:origin",
        "Life Changer": "q:life_changer",
        "Advisor": "q:advisor",
        "Visionary": "q:visionary",
        "Creator": "q:creator",
        "⬅️ Back": "menu:main"
    },

    "dual_income": {
        "Club Income": "q:club_income",
        "Liquidity Pool": "q:lp_benefits",
        "⬅️ Back": "menu:main"
    },

    "others": {
        "Rewards": "q:rewards",
        "Referral": "q:referral",
        "Registration": "q:registration",
        "Airdrop": "q:airdrop",
        "Wallet & Security": "q:wallet_security",
        "Token Price": "q:token_price",
        "Terms & Conditions": "q:terms",
        "Withdrawal Policy": "q:withdrawal",
        "Objectives": "q:objectives",
        "Important Note": "q:important_note",
        "⬅️ Back": "menu:main"
    }
}

def build_menu(menu_key: str):
    menu = MENUS.get(menu_key, {})
    keyboard = []

    for text, callback in menu.items():
        keyboard.append([{
            "text": text,
            "callback_data": callback
        }])

    return {"inline_keyboard": keyboard}


TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

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

ANSWERS = {
    "deposit_slab": "Deposit slabs define fixed investment tiers with benefits.",
    "self_staking": "Stake FINUX tokens to earn passive rewards.",
    "min_deposit": "Minimum deposit ensures platform stability.",

    "mstc": "MSTC funds ecosystem development.",
    "usdce": "USDCe ensures stable-value allocation.",
    "fund_distribution": "Funds are distributed transparently.",

    "origin": "Entry-level rank with basic benefits.",
    "life_changer": "Higher rank with increased rewards.",
    "advisor": "Leadership-based income level.",
    "visionary": "Advanced rank with network rewards.",
    "creator": "Top-tier rank with maximum benefits.",

    "club_income": "Club income rewards active members.",
    "lp_benefits": "Liquidity pools generate passive earnings.",

    "rewards": "Earn rewards via activity and staking.",
    "referral": "Invite users and earn referral income.",
    "registration": "Simple onboarding to FINUX ecosystem.",
    "airdrop": "Free tokens for eligible users.",
    "wallet_security": "Secure wallets protect your assets.",
    "token_price": "Token price depends on market demand.",
    "terms": "Platform rules and conditions apply.",
    "withdrawal": "Withdrawals follow defined policies.",
    "objectives": "FINUX aims for decentralized growth.",
    "important_note": "Always verify before investing."
}

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
async def telegram_webhook(request: Request):
    data = await request.json()
    logging.info(f"TELEGRAM UPDATE: {data}")

    async with httpx.AsyncClient(timeout=15) as client:

        # ---------- CALLBACK ----------
        if "callback_query" in data:
            cq = data["callback_query"]
            chat_id = cq["message"]["chat"]["id"]
            query_id = cq["id"]
            payload = cq.get("data", "")

            # acknowledge callback
            await client.post(
                f"{TELEGRAM_API}/answerCallbackQuery",
                json={"callback_query_id": query_id}
            )

            # MENU navigation
            if payload.startswith("menu:"):
                menu_key = payload.replace("menu:", "")
                await client.post(
                    f"{TELEGRAM_API}/editMessageText",
                    json={
                        "chat_id": chat_id,
                        "message_id": cq["message"]["message_id"],
                        "text": "📌 Please choose:",
                        "reply_markup": build_menu(menu_key)
                    }
                )
                return {"ok": True}

            # QUESTION
            if payload.startswith("q:"):
                key = payload.replace("q:", "")
                answer = ANSWERS.get(key, "Information coming soon.")

                await client.post(
                    f"{TELEGRAM_API}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": answer
                    }
                )
                return {"ok": True}

        # ---------- /start ----------
        message = data.get("message")
        if not message:
            return {"ok": True}

        chat_id = message["chat"]["id"]
        text = message.get("text", "")

        if text == "/start":
            await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": "Welcome to FINUX 🚀\nChoose a category below:",
                    "reply_markup": build_menu("main")
                }
            )

    return {"ok": True}
