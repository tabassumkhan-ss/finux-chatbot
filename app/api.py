import os
import logging
import httpx

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


logging.basicConfig(level=logging.INFO)
logging.info("DATA DIR CONTENTS: %s", os.listdir("data"))

# ===================== TELEGRAM =====================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

WELCOME_TEXT = (
    "✨ *Welcome to FINUX..."
    )

# ===================== MENUS =====================

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
        "⬅️ Back": "menu:main",
    },
    "funds": {
        "MSTC": "q:mstc",
        "USDCe": "q:usdce",
        "⬅️ Back": "menu:main",
    },
    "ranks": {
        "Origin": "q:origin",
        "Life Changer": "q:life_changer",
        "⬅️ Back": "menu:main",
    },
    "dual_income": {
        "Club Income": "q:club_income",
        "⬅️ Back": "menu:main",
    },
    "others": {
        "Airdrop": "q:airdrop",
        "Wallet & Security": "q:wallet_security",
        "⬅️ Back": "menu:main",
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
        [{"text": "🚀 Open App", "url": "https://finux-chatbot-production.up.railway.app"}],
        [
            {"text": "📢 Channel", "url": "https://t.me/FINUX_ADV"},
            {"text": "🌐 Website", "url": "https://finux-chatbot-production.up.railway.app"},
        ],
    ]

def build_menu(menu_key):
    keyboard = header_buttons()
    for label, action in MENUS.get(menu_key, {}).items():
        keyboard.append([{"text": label, "callback_data": action}])
    return {"inline_keyboard": keyboard}

# ===================== FASTAPI =====================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ static folder
app.mount("/static", StaticFiles(directory="data"), name="static")

# ===================== TELEGRAM WEBHOOK =====================

@app.post("/telegram")
async def telegram_webhook(request: Request):
    data = await request.json()
    logging.info(f"TELEGRAM UPDATE: {data}")

    async with httpx.AsyncClient(timeout=30) as client:

        # ---------- CALLBACK ----------
        if "callback_query" in data:
            cq = data["callback_query"]
            chat_id = cq["message"]["chat"]["id"]
            msg_id = cq["message"]["message_id"]
            payload = cq.get("data", "")

            await client.post(
                f"{TELEGRAM_API}/answerCallbackQuery",
                json={"callback_query_id": cq["id"]},
            )

            if payload.startswith("menu:"):
                menu_key = payload.replace("menu:", "")
                await client.post(
                    f"{TELEGRAM_API}/editMessageText",
                    json={
                        "chat_id": chat_id,
                        "message_id": msg_id,
                        "text": WELCOME_TEXT,
                        "reply_markup": build_menu(menu_key),
                    },
                )
                return {"ok": True}

            if payload.startswith("q:"):
                key = payload.replace("q:", "")
                answer = ANSWERS.get(key, "Information coming soon.")

                await client.post(
                    f"{TELEGRAM_API}/editMessageText",
                    json={
                        "chat_id": chat_id,
                        "message_id": msg_id,
                        "text": f"{WELCOME_TEXT}\n\n{answer}",
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

            # 1️⃣ SEND IMAGE (UPLOAD FILE — THIS IS THE KEY)
            image_path = "data/finux.png"
            if os.path.exists(image_path):
                with open(image_path, "rb") as img:
                    await client.post(
                        f"{TELEGRAM_API}/sendPhoto",
                        data={"chat_id": chat_id},
                        files={"photo": img},
                    )

            # 2️⃣ SEND WELCOME MESSAGE
            await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": WELCOME_TEXT,
                    "parse_mode": "Markdown",
                },
            )

            # 3️⃣ SEND MENU
            await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": "👇 *Main Menu*",
                    "parse_mode": "Markdown",
                    "reply_markup": build_menu("main"),
                },
            )

        return {"ok": True}
