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

    # Clean common question words
    stop_words = {"what", "is", "how", "does", "the", "a", "an", "of", "to", "in"}
    keywords = [w for w in question.split() if w not in stop_words and len(w) > 3]

    best_match = ""
    best_score = 0

    for i, line in enumerate(DOCUMENT_TEXT):
        line_l = line.lower()

        score = sum(1 for word in keywords if word in line_l)

        if score > best_score:
            best_score = score
            best_match = line

            # also attach next line for context
            if i + 1 < len(DOCUMENT_TEXT):
                best_match += " " + DOCUMENT_TEXT[i + 1]

    if best_score > 0:
        # return only first 2 sentences max
        sentences = best_match.split(".")
        return ".".join(sentences[:2]).strip() + "."

    return ""

def generate_answer(question: str) -> str:

    # 1️⃣ Try FINUX documents first
    doc_answer = find_short_answer(question)
    if doc_answer:
        return doc_answer

    # 2️⃣ Gemini fallback (very short answer enforced)
    try:
        response = client.models.generate_content(
            model="models/gemini-flash-latest",
            contents=f"Answer in maximum 2 short sentences. No bullet points. No formatting. Question: {question}"
        )

        if response.text:
            return response.text.strip()

    except Exception as e:
        logging.error(f"Gemini error: {e}")

    return "Sorry, I could not generate a response."

# ================ TELEGRAM ===============

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ================= GEMINI =================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

client = genai.Client(api_key=GEMINI_API_KEY)

# ===================== MENUS =====================
MAIN_MENU = {
    "💼 Wallet": "menu:wallet",
    "💰 Deposit": "menu:deposit",
    "🪙 Minting": "menu:minting",
    "📦 Others": "menu:others",
}

OTHERS_MENU = {
    "💧 Liquidity Pool": "menu:lp",
    "🔐 FNX Self Staking": "menu:staking",
    "💸 Withdraw": "menu:withdraw",
    "🎁 Airdrop": "menu:airdrop",
    "🤝 Affiliate Program": "menu:affiliate",
    "🏅 Ranks & Clubs": "menu:ranks",
    "💎 Triple Income System": "menu:triple_income",
    "📜 Terms & Conditions": "q:terms_conditions",
    "⚠ Risk Disclaimer": "q:risk_disclaimer",
    "⬅ Back": "menu:main",
}

WALLET_MENU = {
    "What is a Wallet?": "q:wallet_info",
    "How to Create a Wallet?": "q:wallet_create",
    "Wallet Security": "q:wallet_security",
    "Private Key / Seed Phrase": "q:wallet_private",
    "⬅️ Back to Main": "menu:main",
}

DEPOSIT_MENU = {
    "💰 Minimum Deposit": "q:deposit_min",
    "📊 Accepted Deposit Plans": "q:deposit_plans",
    "📦 Deposit Structure": "q:deposit_structure",
    "⛓ Blockchain": "q:deposit_blockchain",
    "⬅ Back": "menu:main",
}

MINTING_MENU = {
    "⚙️ What is Minting?": "q:minting_info",
    "⏱ When Minting Happens?": "q:minting_time",
    "📍 Minted Token Location": "q:minting_location",
    "⬅ Back": "menu:main",
}

LP_MENU = {
    "💧 What is Liquidity Pool?": "q:lp_info",
    "🔗 LP Pair": "q:lp_pair",
    "⭐ Benefits of LP": "q:lp_benefits",
    "💰 LP Rewards": "q:lp_rewards",
    "⬅ Back": "menu:others",
}

STAKING_MENU = {
    "🔐 What is Staking?": "q:staking_info",
    "⚙ How Staking Works": "q:staking_work",
    "💰 Rewards from Staking": "q:staking_rewards",
    "⬅ Back": "menu:others",
}

WITHDRAW_MENU = {
    "💸 Can I withdraw anytime?": "q:withdraw_anytime",
    "💰 Withdrawal currency": "q:withdraw_currency",
    "🔥 Token burning mechanism": "q:withdraw_burn",
    "⬅ Back": "menu:others",
}

AIRDROP_MENU = {
    "🎁 Airdrop eligibility": "q:airdrop_eligibility",
    "💰 Airdrop reward": "q:airdrop_reward",
    "📋 Conditions": "q:airdrop_conditions",
    "⬅ Back": "menu:others",
}

AFFILIATE_MENU = {
    "🤝 What is the affiliate program": "q:affiliate_info",
    "👥 Team business": "q:affiliate_team",
    "📈 Why affiliate is important": "q:affiliate_importance",
    "⬅ Back": "menu:others",
}

RANKS_MENU = {
    "🏅 Rank Structure": "q:rank_structure",
    "🎖 Club rewards": "q:club_rewards",
    "📊 Rank requirements": "q:rank_requirements",
    "⬅ Back": "menu:others",
}

TRIPLE_MENU = {
    "💎 What is the Triple Income System": "q:triple_info",
    "📉 What happens after reaching the limit": "q:triple_limit",
    "⬅ Back": "menu:others",
}



MENUS = {
    "main": MAIN_MENU,
    "wallet": WALLET_MENU,
    "deposit": DEPOSIT_MENU,
    "minting": MINTING_MENU,
    "others": OTHERS_MENU,
    "lp": LP_MENU,
    "staking": STAKING_MENU,
    "withdraw": WITHDRAW_MENU,
    "airdrop": AIRDROP_MENU,
    "affiliate": AFFILIATE_MENU,
    "ranks": RANKS_MENU,
    "triple_income": TRIPLE_MENU,
}

HARDCODED_ANSWERS = {
    "wallet_info": "A FINUX wallet is a digital wallet where your *FNX tokens and rewards* are stored.\nIt is *automatically generated* when you register in the system.",

"wallet_create": "1️⃣ Download the wallet from the official website:\nhttps://finux.online\n2️⃣ Your wallet will be generated automatically.\n⚠️ Secure your *private key / seed phrase*.\n3️⃣ Sign up on DEX to start using your wallet.",

"wallet_security": "FINUX wallets operate in a secure blockchain environment.\nHowever, users must protect their *private key or seed phrase*.\n⚠️ If you lose it, the company *cannot recover your funds*.",

"wallet_private": "Your private key or seed phrase is a *secret code* that gives access to your wallet.\n⚠️ Never share it with anyone.\nAnyone with this key can *control your funds*.",
    
"deposit_min": "💰 *Minimum Deposit*\nThe minimum deposit is *$20*.",

"deposit_plans": "📊 *Accepted Deposit Plans*\nYou can deposit:\n• $20\n• $50\n• $100\n• $200\n• Multiples of $100",

"deposit_structure": "📦 *Deposit Structure*\nYour deposit is split into:\n• 30% MSTC\n• 70% USDC (Polygon Network)",

"deposit_blockchain": "⛓ *Blockchain*\nThe system uses *MEP-20 blockchain contract*.",    
    
"minting_info": "Minting means creating a new FNX token in the system.",

"minting_time": "After your deposit transaction is completed.",

"minting_location": "The system automatically credits the minted FNX token to your wallet.",    
    
    "lp_info": "A Liquidity Pool is where users provide tokens to help trading happen smoothly.",

"lp_pair": "FNX + USDC pair is used.",

"lp_benefits": "Stable trading\n• Daily passive income\n• High rewards\n• Community growth\n• Strong ecosystem support",

"lp_rewards": "You can earn daily rewards up to *5% MPY (Monthly Percentage Yield)*.\nThese rewards are generated from the system's trading and ecosystem activity.",
    
"staking_info": "🔐 *What is Staking?*\nIt means locking FNX tokens in the system to earn rewards.",

"staking_work": "⚙ *How Staking Works*\nThe staking process is very simple:\n• Deposit funds into the platform\n• FNX tokens are minted and credited to your wallet\n• Stake your FNX tokens in the Self-Staking section\n• The system generates daily rewards automatically\n• You can withdraw rewards anytime",

"staking_rewards": "💰 *Rewards from Staking*\nUp to *2% MPY (Monthly Percentage Yield)* daily reward.",    
    
"withdraw_anytime": "Yes, FNX rewards can be withdrawn anytime.",

"withdraw_currency": "You will receive *USDC* in your wallet instantly.",

"withdraw_burn": "When you withdraw FNX:\n• 50% FNX is burned\n• 50% FNX goes back to supply.\nThis helps control token supply.",    
    
"airdrop_eligibility": "Yes, you must have at least *5 direct paid referrals*.",

"airdrop_reward": "You receive *50 FNX tokens*.",

"airdrop_conditions": "• Wallet must be registered\n• User must be verified\n• Duplicate referrals are not counted",   
    
"affiliate_info": "It is a referral program where you earn rewards by building a team.",

"affiliate_team": "The total deposits made by your team.",

"affiliate_importance": "It helps grow the community and increases earnings.",    
    
"rank_structure": "• Rank 1 — Origin\n• Rank 2 — Life Changer\n• Rank 3 — Advisor\n• Rank 4 — Visionary\n• Rank 5 — Creator",

"club_rewards": "• Rank 1 (Origin) — 10%\n• Rank 2 (Life Changer) — 16% (3% CTO club share)\n• Rank 3 (Advisor) — 20% (2.5% CTO club share)\n• Rank 4 (Visionary) — 23% (2% CTO club share)\n• Rank 5 (Creator) — 25% (1.5% CTO club share)",

"rank_requirements": "• Rank 1 (Origin)\n  • Self activation\n• Rank 2 (Life Changer)\n  • $1000 team business\n  • 10 active origins\n  • Minimum $30 LP\n• Rank 3 (Advisor)\n  • $5000 team business\n  • 2 active life changers\n  • Minimum $100 LP\n• Rank 4 (Visionary)\n  • $25,000 team business\n  • 2 active advisors\n  • Minimum $300 LP\n• Rank 5 (Creator)\n  • $100,000 team business\n  • 2 active visionaries\n  • Minimum $1000 LP",    
    
"triple_info": "Users can earn from three sources:\n• Performance income — up to 3x\n• Liquidity pool reward — up to 3x\n• FNX staking — up to 2x",

"triple_limit": "After *3x performance income*, you must *retop-up* to continue earning.",    
    
"terms_conditions": "*General T&C*\n• Anyone can join the program\n• Rewards based on company policy\n• Company may update program anytime\n\n*Airdrop T&C*\n• Wallet registration + verification\n• Limited period\n• Duplicate referrals not counted\n\n*Additional T&C*\n• LP 50% counts in team business\n• Performance income limit: 3X\n• Retop-up required after limit\n• Retop-up gives 50% FNX",   
    
"risk_disclaimer": "• Crypto investments carry risk\n• Earnings are not guaranteed\n• Users must secure their wallets\n• Company is not responsible for lost private keys",    
   
 }

# ===================== UI HELPERS =====================

def header_buttons():
    return [
        [{"text": " Open App", "url": "https://finux-chatbot-production.up.railway.app"}],
        [
            {"text": "Channel", "url": "https://t.me/Finuxofficiallive"},
            {"text": " Website", "url": "https://finux.online/"},
        ],
    ]

def build_menu(menu_key):
    keyboard = header_buttons()

    menu_items = list(MENUS.get(menu_key, {}).items())

    row = []
    for label, action in menu_items:
        row.append({"text": label, "callback_data": action})

        if len(row) == 2:
            keyboard.append(row)
            row = []

    if row:
        keyboard.append(row)

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

                
                
                message_id = cq["message"]["message_id"]

                await client.post(
                f"{TELEGRAM_API}/editMessageText",
                 json={
                "chat_id": chat_id,
                "message_id": message_id,
                "text": "🚀 *FINUX Assistant*\nPlease choose an option:",
"parse_mode": "Markdown",
                 "reply_markup": build_menu(menu_key),
    },
)
                return {"ok": True}

            # DOCUMENT / HARDCODED SEARCH
            if payload.startswith("q:"):

                key = payload.replace("q:", "")

                # 1️⃣ Hardcoded answers first
                answer = HARDCODED_ANSWERS.get(key)

                # 2️⃣ Document search
                if not answer:
                    topic = key.replace("_", " ")
                    answer = find_short_answer(topic)

                # 3️⃣ Gemini fallback
                if not answer:
                    topic = key.replace("_", " ")
                    answer = generate_answer(topic)

                message_id = cq["message"]["message_id"]

                # decide which menu to show after answer
                menu_to_show = "main"

                if key.startswith("wallet"):
                 menu_to_show = "wallet"
                elif key.startswith("deposit"): 
                 menu_to_show = "deposit"
                elif key.startswith("minting"):
                 menu_to_show = "minting" 
                elif key.startswith("lp"):
                 menu_to_show = "lp"
                elif key.startswith("staking"):
                 menu_to_show = "staking"
                elif key.startswith("withdraw"):
                 menu_to_show = "withdraw"
                elif key.startswith("airdrop"):
                 menu_to_show = "airdrop"
                elif key.startswith("affiliate"):
                 menu_to_show = "affiliate"
                elif key.startswith("rank"):
                 menu_to_show = "ranks"
                elif key.startswith("triple"):
                 menu_to_show = "triple_income"

                await client.post(
                  f"{TELEGRAM_API}/editMessageText",
                   json={
                    "chat_id": chat_id,
                     "message_id": message_id,
                     "text": answer,
                     "parse_mode": "Markdown",
                     "reply_markup": build_menu(menu_to_show),
    },
)

                # Save to DB
                try:
                    save_chat(
                        "telegram",
                        str(chat_id),
                        "",
                        key,
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
                    "text": "🚀 *FINUX Assistant*\nPlease choose an option:",
"parse_mode": "Markdown",
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