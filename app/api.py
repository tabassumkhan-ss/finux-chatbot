import os
import logging
import httpx
from google import genai
from google.genai import types
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from docx import Document
from pypdf import PdfReader
from app.db import save_chat

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
    if t.strip()
]

DOCUMENT_TEXT = load_documents()

def find_short_answer(question: str) -> str:
    question = question.lower().strip()

    matched_lines = []

    for i, line in enumerate(DOCUMENT_TEXT):
        line_l = line.lower()

        # Direct match
        if question in line_l:
            matched_lines.append(line)

            # Also grab next 2 lines for context
            if i + 1 < len(DOCUMENT_TEXT):
                matched_lines.append(DOCUMENT_TEXT[i + 1])
            if i + 2 < len(DOCUMENT_TEXT):
                matched_lines.append(DOCUMENT_TEXT[i + 2])

            break

    if matched_lines:
        return " ".join(matched_lines)[:400]

    return "Information not available in FINUX documents."

def generate_answer(question: str) -> str:

    # 1️⃣ Try FINUX docs first
    doc_answer = find_short_answer(question)

    if doc_answer != "Information not available in FINUX documents.":
        return doc_answer

    # 2️⃣ Gemini fallback
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash-002",
            contents=question
        )

        if response and response.text:
            return response.text.strip()

    except Exception as e:
        logging.error(f"Gemini error: {e}")

    return "Sorry, I could not generate a response."

logging.info("Loaded %d lines from FINUX documents", len(DOCUMENT_TEXT))

if os.path.exists(DATA_DIR):
    logging.info("DATA DIR CONTENTS: %s", os.listdir(DATA_DIR))
else:
    logging.warning("DATA directory does not exist")

# ================ TELEGRAM ===============

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ================= GEMINI =================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

client = genai.Client(api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(api_version="v1"))


# ===================== MENUS =====================
MAIN_MENU = {
    "🆕 Create Wallet": "q:create_wallet",
    "💰 Deposit": "q:deposit",
    "🪙 Minting": "q:minting",
    "📦 Others": "menu:others",
}

OTHERS_MENU = {
    "💧 Liquidity Pool": "q:liquidity_pool",
    "🔐 FNX Self-Staking": "q:self_staking",
    "💸 Withdraw": "q:withdraw",
    "🎁 Airdrop": "q:airdrop",
    "🤝 Affiliate Program": "q:affiliate",
    "🏅 Ranks & Clubs": "q:ranks_clubs",
    "💎 Triple Income System": "q:triple_income",
    "📜 Terms & Conditions": "q:terms",
    "❓ FAQ": "q:faq",
    "🛠 Support": "q:support",
    "⬅️ Back to Main": "menu:main",
}

MENUS = {
    "main": MAIN_MENU,
    "others": OTHERS_MENU,
}

# ===================== UI HELPERS =====================

def header_buttons():
    return [
        [{"text": " Open App", "url": "https://finux-chatbot-production.up.railway.app"}],
        [
            {"text": "Channel", "url": "https://t.me/Finuxofficiallive"},
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

    answer = generate_answer(question)

    # ✅ Save to DB
    try:
        save_chat(
            "web",
            "web_user",
            "",
            question,
            answer
        )
    except Exception as e:
        logging.error(f"DB save error (web): {e}")

    return {"response": answer}

# ✅ static folder
app.mount("/static", StaticFiles(directory="data"), name="static")

@app.get("/post-button")
async def post_button():
    channel_username = "@Finuxofficiallive"

    async with httpx.AsyncClient() as client:

        # Get chat info
        chat_info = await client.post(
            f"{TELEGRAM_API}/getChat",
            json={"chat_id": channel_username}
        )

        chat_data = chat_info.json()

        # Delete old pinned message if exists
        if "pinned_message" in chat_data.get("result", {}):
            old_message_id = chat_data["result"]["pinned_message"]["message_id"]

            await client.post(
                f"{TELEGRAM_API}/deleteMessage",
                json={
                    "chat_id": channel_username,
                    "message_id": old_message_id
                }
            )

        # Send new button
        send_response = await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": channel_username,
                "text": "Welcome to FINUX Chat Bot",
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

        send_result = send_response.json()
        new_message_id = send_result["result"]["message_id"]

        # Pin new message
        await client.post(
            f"{TELEGRAM_API}/pinChatMessage",
            json={
                "chat_id": channel_username,
                "message_id": new_message_id,
                "disable_notification": True
            },
        )

    return {"status": "Pinned button refreshed cleanly"}

@app.get("/check-admin")
async def check_admin():
    channel_username = "@Finuxofficiallive"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TELEGRAM_API}/getChatMember",
            json={
                "chat_id": channel_username,
                "user_id": 8579775227
            },
        )

    return response.json()


# ===================== TELEGRAM WEBHOOK =====================

@app.post("/telegram")
async def telegram_webhook(request: Request):
    data = await request.json()
    logging.info(f"TELEGRAM UPDATE: {data}")

    async with httpx.AsyncClient(timeout=30) as client:

        # ================= CALLBACK HANDLER =================
        if "callback_query" in data:
            cq = data["callback_query"]
            chat_id = cq["message"]["chat"]["id"]
            payload = cq.get("data", "")

            # acknowledge callback
            await client.post(
                f"{TELEGRAM_API}/answerCallbackQuery",
                json={"callback_query_id": cq["id"]},
            )

            # MENU navigation
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

            # DOCUMENT SEARCH from button
            if payload.startswith("q:"):
                topic = payload.replace("q:", "").replace("_", " ")
                answer = answer = find_short_answer(topic)

                await client.post(
                    f"{TELEGRAM_API}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": answer,
                        "reply_markup": build_menu("main"),
                    },
                )

                # Save to DB
                try:
                    save_chat(
                        "telegram",
                        str(chat_id),
                        "",
                        topic,
                        answer
                    )
                except Exception as e:
                    logging.error(f"DB save error (callback): {e}")

                return {"ok": True}

        # ================= NORMAL MESSAGE =================
        message = data.get("message")
        if not message:
            return {"ok": True}

        chat_id = message["chat"]["id"]
        text = message.get("text", "").strip()

        # /start command
        if text.startswith("/start"):

            image_path = os.path.join(DATA_DIR, "finux.png")
            if os.path.exists(image_path):
                with open(image_path, "rb") as img:
                    await client.post(
                        f"{TELEGRAM_API}/sendPhoto",
                        data={"chat_id": chat_id},
                        files={"photo": img},
                    )

            await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": "👇 Choose an option",
                    "reply_markup": build_menu("main"),
                },
            )
            return {"ok": True}

        # USER typed question
        if text:
            answer = generate_answer(text)

            await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": answer,
                },
            )

            # Save to DB
            try:
                save_chat(
                    "telegram",
                    str(chat_id),
                    message.get("from", {}).get("username", ""),
                    text,
                    answer
                )
            except Exception as e:
                logging.error(f"DB save error (telegram): {e}")

        return {"ok": True}