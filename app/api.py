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

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
WELCOME_TEXT = (
    "✨ *Welcome to FINUX*\n\n"
    "Decentralized blockchain + AI ecosystem.\n\n"
    "Choose an option below 👇"
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------
app = FastAPI()

# -------------------------------------------------
# STATIC FILES
# -------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# MENUS
# -------------------------------------------------
MENUS = {
    "main": {
        "💰 Deposit": "menu:deposit",
        "📊 Fund Distribution": "menu:funds",
        "🏆 Rank Wise Rewards": "menu:ranks",
        "🔁 Dual Income System": "menu:dual_income",
        "📌 Others": "menu:others",
    },

    "deposit": {
        "Deposit Slab": "q:deposit_slab",
        "Self-Staking": "q:self_staking",
        "Min. Deposit & Structure": "q:min_deposit",
        "⬅️ Back": "menu:main",
    },

    "funds": {
        "MSTC": "q:mstc",
        "USDCe": "q:usdce",
        "Fund Distribution": "q:fund_distribution",
        "⬅️ Back": "menu:main",
    },

    "ranks": {
        "Origin": "q:origin",
        "Life Changer": "q:life_changer",
        "Advisor": "q:advisor",
        "Visionary": "q:visionary",
        "Creator": "q:creator",
        "⬅️ Back": "menu:main",
    },

    "dual_income": {
        "Club Income": "q:club_income",
        "Liquidity Pool": "q:lp_benefits",
        "⬅️ Back": "menu:main",
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
        "⬅️ Back": "menu:main",
    },
}

# -------------------------------------------------
# ANSWERS (SHORT)
# -------------------------------------------------
ANSWERS = {
    "deposit_slab": "Fixed deposit tiers with defined benefits.",
    "self_staking": "Stake FINUX to earn passive rewards.",
    "min_deposit": "Minimum deposit ensures system stability.",

    "mstc": "MSTC funds ecosystem operations.",
    "usdce": "USDCe maintains stable fund allocation.",
    "fund_distribution": "Funds are distributed transparently.",

    "origin": "Entry-level rank with base rewards.",
    "life_changer": "Mid-level rank with higher income.",
    "advisor": "Leadership rank with network rewards.",
    "visionary": "Advanced rank with global benefits.",
    "creator": "Top-tier rank with maximum income.",

    "club_income": "Club income rewards active members.",
    "lp_benefits": "Liquidity pools generate passive earnings.",

    "rewards": "Earn rewards via staking & activity.",
    "referral": "Invite users and earn commissions.",
    "registration": "Simple onboarding into FINUX.",
    "airdrop": "Free tokens for eligible users.",
    "wallet_security": "Secure wallets protect assets.",
    "token_price": "Price depends on market demand.",
    "terms": "Platform rules & conditions apply.",
    "withdrawal": "Withdrawals follow set policies.",
    "objectives": "FINUX focuses on decentralized growth.",
    "important_note": "Always verify before investing.",
}

# -------------------------------------------------
# TELEGRAM UI HELPERS
# -------------------------------------------------
def build_header_buttons():
    return [
        [{"text": "🚀 Open App", "url": "https://finux-chatbot-production.up.railway.app"}],
        [
            {"text": "📢 Channel", "url": "https://t.me/FINUX_ADV"},
            {"text": "🌐 Website", "url": "https://finux-chatbot-production.up.railway.app"},
        ],
    ]


def build_menu(menu_key: str):
    keyboard = build_header_buttons()
    for label, action in MENUS.get(menu_key, {}).items():
        keyboard.append([{"text": label, "callback_data": action}])
    return {"inline_keyboard": keyboard}


def send_start_photo(chat_id: int):
    requests.post(
        f"{TELEGRAM_API}/sendPhoto",
        json={
            "chat_id": chat_id,
            "photo": "https://finux-chatbot-production.up.railway.app/static/finux.png",
            "caption": WELCOME_TEXT,
            "parse_mode": "Markdown",
            "reply_markup": build_menu("main"),
        },
    )

# -------------------------------------------------
# LOAD FINUX DOCS (PDF + DOCX)
# -------------------------------------------------
documents = []

pdf_path = os.path.join(RAW_DIR, "finux.pdf")
docx_path = os.path.join(RAW_DIR, "finux.docx")

if os.path.exists(pdf_path):
    documents += PyPDFLoader(pdf_path).load()

if os.path.exists(docx_path):
    documents += Docx2txtLoader(docx_path).load()

if not documents:
    documents = [Document(page_content="FINUX is a decentralized crypto ecosystem.")]

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)
texts = [c.page_content for c in chunks]

db = create_vector_store(texts)

# -------------------------------------------------
# MODELS
# -------------------------------------------------
class ChatRequest(BaseModel):
    message: str

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    with open(os.path.join(BASE_DIR, "ui.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/telegram")
async def telegram_webhook(request: Request):
    data = await request.json()
    logging.info(f"TELEGRAM UPDATE: {data}")

    # ---------- CALLBACK ----------
    if "callback_query" in data:
        cq = data["callback_query"]
        chat_id = cq["message"]["chat"]["id"]
        message_id = cq["message"]["message_id"]
        payload = cq.get("data", "")

        requests.post(
            f"{TELEGRAM_API}/answerCallbackQuery",
            json={"callback_query_id": cq["id"]}
        )

        if payload.startswith("menu:"):
            menu_key = payload.replace("menu:", "")
            requests.post(
                f"{TELEGRAM_API}/editMessageCaption",
                json={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "caption": WELCOME_TEXT,
                    "parse_mode": "Markdown",
                    "reply_markup": build_menu(menu_key),
                },
            )
            return {"ok": True}

        if payload.startswith("q:"):
            key = payload.replace("q:", "")
            answer = ANSWERS.get(key, "Information coming soon.")

            requests.post(
                f"{TELEGRAM_API}/editMessageCaption",
                json={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "caption": f"{WELCOME_TEXT}\n\n*Answer:*\n{answer}",
                    "parse_mode": "Markdown",
                    "reply_markup": build_menu("main"),
                },
            )
            return {"ok": True}

    # ---------- MESSAGE ----------
    message = data.get("message")
    if not message:
        return {"ok": True}

    chat_id = message["chat"]["id"]
    text = message.get("text", "")

    if text == "/start":
        send_start_photo(chat_id)
        return {"ok": True}

    return {"ok": True}
