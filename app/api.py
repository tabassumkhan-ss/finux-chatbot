import os
import logging
import httpx

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from docx import Document
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)

class ChatRequest(BaseModel):
    message: str

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ===================== DOCUMENT LOADER =====================

DOCUMENT_TEXT = []

def load_documents():
    texts = []

    # PDF
    pdf_path = os.path.join(DATA_DIR, "finux.pdf")
    if os.path.exists(pdf_path):
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.extend(text.split("\n"))

    # DOCX
    docx_path = os.path.join(DATA_DIR, "finux.docx")
    if os.path.exists(docx_path):
        doc = Document(docx_path)
        for para in doc.paragraphs:
            if para.text.strip():
                texts.append(para.text)

    return [
    t.strip()
    for t in texts
    if len(t.strip()) > 30 and "•" not in t
]

DOCUMENT_TEXT = load_documents()

def find_short_answer(question: str) -> str:
    q_words = [w for w in question.lower().split() if len(w) > 3]

    for line in DOCUMENT_TEXT:
        line_l = line.lower()

        # must match at least one meaningful keyword
        if any(w in line_l for w in q_words):
            # return only first sentence, very short
            return line.split(".")[0].strip()[:160]

    return "Information not available in FINUX documents."

logging.info("Loaded %d lines from FINUX documents", len(DOCUMENT_TEXT))

if os.path.exists(DATA_DIR):
    logging.info("DATA DIR CONTENTS: %s", os.listdir(DATA_DIR))
else:
    logging.warning("DATA directory does not exist")

# ===================== TELEGRAM =====================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


# ===================== MENUS =====================

MAIN_MENU = {
    "💰 Deposit": "menu:deposit",
    "🏆 Rewards": "menu:rewards",
    "👥 Referral": "menu:referral",
    "📦 Others": "menu:others",
}

OTHERS_MENU = {
    "📈 Fund Distribution": "menu:funds",
    "🏅 Rank Wise Rewards": "menu:ranks",
    "🔁 Dual Income System": "menu:dual_income",
    "🏦 Club Income": "menu:clubincome",
    "🪙 Token Price": "menu:token",
    "💸 Withdrawal Policy": "menu:withdraw",
    "📜 Terms & Conditions": "menu:terms",
    "🔐 Wallet & Security": "menu:wallet",
    "🚀 FINUX Mining Project": "menu:fmp",
    "⬅️ Back to Main": "menu:main",
}

MENUS = {
    "main": MAIN_MENU,
    "others": OTHERS_MENU,

    "deposit": {
        "Deposit Slab": "q:deposit_slab",
        "Self-Staking": "q:self_staking",
        "⬅️ Back": "menu:main",
    },

    "rewards": {
        "Rank Rewards": "q:rank_rewards",
        "Club Rewards": "q:club_rewards",
        "⬅️ Back": "menu:main",
    },

        "referral": {
        "How Referral Works": "q:referral",
        "Referral Income": "q:referral_income",
        "⬅️ Back": "menu:main",
    },

        "clubincome": {
        "Club Income Info": "q:club_income",
        "⬅️ Back": "menu:others",
    },
}

ANSWERS = {
    "deposit_slab": "Fixed investment tiers with benefits.",
    "self_staking": "Stake FINUX to earn rewards.",
    "mstc": "MSTC supports ecosystem growth.",
    "usdce": "USDCe ensures stability.",
    "origin": "Entry-level rank.",
    "life_changer": "Higher rewards tier.",
    "club_income": "Income from club activity.",
    "airdrop": "Free tokens for users.",
    "wallet_security": "Your assets are secured.",
}

# ===================== UI HELPERS =====================

def header_buttons():
    return [
        [{"text": " Open App", "url": "https://finux-chatbot-production.up.railway.app"}],
        [
            {"text": "Channel", "url": "https://t.me/FINUX_ADV"},
            {"text": " Website", "url": "https://finux-chatbot-production.up.railway.app"},
        ],
    ]

def build_menu(menu_key):
    keyboard = header_buttons()
    for label, action in MENUS.get(menu_key, {}).items():
        keyboard.append([{"text": label, "callback_data": action}])
    return {"inline_keyboard": keyboard}

# ===================== FASTAPI =====================

app = FastAPI()

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(DATA_DIR, "ui.html"))


@app.post("/chat")
async def chat_api(payload: ChatRequest):
    question = payload.message.strip()

    if not question:
        return {"response": "Please ask a question."}

    answer = find_short_answer(question)
    return {"response": answer}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ static folder
app.mount("/static", StaticFiles(directory="data"), name="static")

@app.get("/post-button")
async def post_button():
    channel_username = "@Finuxofficiallive"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": channel_username,
                "text": " ",  # invisible text
                "reply_markup": {
                    "inline_keyboard": [
                        [
                            {
                                "text": "🚀 Open FINUX Chat Bot",
                                "url": "https://t.me/finuxchatbot?start=channel"
                            }
                        ]
                    ]
                }
            },
        )

    return response.json()


# ===================== TELEGRAM WEBHOOK =====================

@app.post("/telegram")
async def telegram_webhook(request: Request):
    data = await request.json()
    logging.info(f"TELEGRAM UPDATE: {data}")

    async with httpx.AsyncClient(timeout=30) as client:

        # ---------- CALLBACK (MENU BUTTONS) ----------
        if "callback_query" in data:
            cq = data["callback_query"]
            chat_id = cq["message"]["chat"]["id"]
            payload = cq.get("data", "")

            # acknowledge callback
            await client.post(
                f"{TELEGRAM_API}/answerCallbackQuery",
                json={"callback_query_id": cq["id"]},
            )

            # MENU HANDLER (send as chat message)
            if payload.startswith("menu:"):
                menu_key = payload.replace("menu:", "")
                await client.post(
                    f"{TELEGRAM_API}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": "👇 Choose an option",
                        "reply_markup": build_menu(menu_key),
                    },
                )
                return {"ok": True}

            # INFO HANDLER (send as chat message)
            if payload.startswith("q:"):
                key = payload.replace("q:", "")
                answer = find_short_answer(key.replace("_", " "))

                await client.post(
                    f"{TELEGRAM_API}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": answer,
                        "reply_markup": build_menu("main"),
                    },
                )
                return {"ok": True}

        # ---------- NORMAL MESSAGE ----------
        message = data.get("message")
        if not message:
            return {"ok": True}

        chat_id = message["chat"]["id"]
        text = message.get("text", "").strip()

        # /start command
        if text == "/start":

            # 1️⃣ Send image (optional)
            image_path = os.path.join(DATA_DIR, "finux.png")
            if os.path.exists(image_path):
                with open(image_path, "rb") as img:
                    await client.post(
                        f"{TELEGRAM_API}/sendPhoto",
                        data={"chat_id": chat_id},
                        files={"photo": img},
                    )

            # 2️⃣ Send menu
            await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": "👇 Choose an option  to Explore Finux                            ",
                    "reply_markup": build_menu("main"),
                },
            )
            return {"ok": True}

        # ---------- USER TYPED QUESTION ----------
        if text:
            answer = find_short_answer(text)

            await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": answer,
                },
            )

        return {"ok": True}
