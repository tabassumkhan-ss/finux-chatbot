import logging
import os
from dotenv import load_dotenv

load_dotenv()   # load env FIRST

logging.basicConfig(level=logging.INFO)

import httpx
import faiss
import numpy as np
from google import genai
from fastapi import FastAPI, Request
from app.db import get_conn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.db import save_chat, init_db

class ChatRequest(BaseModel):
    message: str
    session_id: str

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ===================== DOCUMENT LOADER =====================

DOCUMENT_TEXT = []

def load_documents():
    texts = []

    # PDF loader
    pdf_path = os.path.join(DATA_DIR, "finux.pdf")
    if os.path.exists(pdf_path):
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                paragraphs = text.split("\n\n")
                for p in paragraphs:
                    if p.strip():
                        texts.append(p.strip())

    # DOCX loader
    docx_path = os.path.join(DATA_DIR, "finux.docx")
    if os.path.exists(docx_path):
        doc = Document(docx_path)

        chunk = ""

        for para in doc.paragraphs:

            text = para.text.strip()

            if not text:
                continue

            chunk += text + " "

            if len(chunk) > 300:
                texts.append(chunk.strip())
                chunk = ""

        if chunk:
            texts.append(chunk.strip())

    return [
        t.strip()
        for t in texts
        if t.strip()
    ]

DOCUMENT_TEXT = load_documents()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embedding_model.encode(DOCUMENT_TEXT)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))


def semantic_search(question: str, top_k=2):

    question_vector = embedding_model.encode([question])

    distances, indices = index.search(np.array(question_vector), top_k)

    results = []

    for idx in indices[0]:
        if idx < len(DOCUMENT_TEXT):
            results.append(DOCUMENT_TEXT[idx])

    return "\n".join(results)

def detect_intent(question: str):

    q = question.lower()

    finux_keywords = [
        "finux",
        "rank",
        "origin",
        "life changer",
        "advisor",
        "visionary",
        "creator",
        "staking",
        "liquidity",
        "lp",
        "withdraw",
        "deposit",
        "minting",
        "referral",
        "club",
        "income",
        "reward"
    ]

    for word in finux_keywords:
        if word in q:
            return "finux"

    return "general"

def generate_answer(question: str):

    intent = detect_intent(question)

    # FINUX question
    if intent == "finux":

        context = semantic_search(question)

        if not context:
         context = "FINUX is a decentralized ecosystem that offers staking, liquidity pools, referral rewards, and club income based on rank achievements."

        prompt = f"""
You are the official FINUX assistant.

Use the provided context to answer the question.

Context:
{context}

Rules:
- Answer in 1–2 short sentences
- Always prioritize the provided FINUX context
- Never invent FINUX rules
- Never say information is missing if it appears in the context


Question:
{question}
"""

    else:

        prompt = f"""
Answer the question in maximum 2 short sentences.

Question:
{question}
"""

    try:

        response = client.models.generate_content(
            model="models/gemini-flash-latest",
            contents=prompt
        )
 
        if response.text:
            return response.text.strip()

    except Exception as e:
        logging.error(f"Gemini error: {e}")

    return "Sorry, I could not generate a response."

def translate(text, lang):
    if not text:
        return text

    if lang == "en":
        return text

    # 🔥 create cache key
    cache_key = f"{lang}:{text}"

    # ✅ return from cache if exists
    if cache_key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cache_key]

    try:
        prompt = f"Translate to {lang}: {text}"

        response = client.models.generate_content(
            model="models/gemini-flash-latest",
            contents=prompt
        )

        if response.text:
            translated = response.text.strip()

            # 🔥 store in cache
            TRANSLATION_CACHE[cache_key] = translated

            return translated

    except Exception as e:
        logging.error(f"Translation error: {e}")

    return text


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
MAIN_MENU = [
    {
        "label": {
            "en": "💼 Wallet",
            "hi": "💼 वॉलेट",
            "mr": "💼 बटवा",
            "bn": "💼 ওয়ালেট"
        },
        "action": "menu:wallet"
    },
    {
        "label": {
            "en": "💰 Deposit",
            "hi": "💰 जमा",
            "mr": "💰 जमा",
            "bn": "💰 জমা"
        },
        "action": "menu:deposit"
    },
    {
        "label": {
            "en": "🪙 Minting",
            "hi": "🪙 मिंटिंग",
            "mr": "🪙 टांकन",
            "bn": "🪙 মিন্টিং"
        },
        "action": "menu:minting"
    },
    {
        "label": {
            "en": "📦 Others",
            "hi": "📦 अन्य",
            "mr": "📦 इतर",
            "bn": "📦 অন্যান্য"
        },
        "action": "menu:others"
    }
]

OTHERS_MENU = [
    {
        "label": {
            "en": "💧 Liquidity Pool",
            "hi": "💧 लिक्विडिटी पूल"
        },
        "action": "menu:lp"
    },
    {
        "label": {
            "en": "🔐 FNX Self Staking",
            "hi": "🔐 FNX स्टेकिंग"
        },
        "action": "menu:staking"
    }
]

WALLET_MENU = [
    {"label": {"en": "What is Wallet?", "hi": "वॉलेट क्या है?"}, "action": "q:wallet_info"},
    {"label": {"en": "Create Wallet", "hi": "वॉलेट बनाएं"}, "action": "q:wallet_create"},
    {"label": {"en": "Wallet Security", "hi": "सुरक्षा"}, "action": "q:wallet_security"},
    {"label": {"en": "Private Key", "hi": "प्राइवेट की"}, "action": "q:wallet_private"},
    {"label": {"en": "⬅ Back", "hi": "⬅ वापस"}, "action": "menu:main"},
]

DEPOSIT_MENU = [
    {"label": {"en": "💰 Minimum Deposit", "hi": "💰 न्यूनतम जमा"}, "action": "q:deposit_min"},
    {"label": {"en": "📊 Accepted Deposit Plans", "hi": "📊 जमा योजनाएं"}, "action": "q:deposit_plans"},
    {"label": {"en": "📦 Deposit Structure", "hi": "📦 जमा संरचना"}, "action": "q:deposit_structure"},
    {"label": {"en": "⛓ Blockchain", "hi": "⛓ ब्लॉकचेन"}, "action": "q:deposit_blockchain"},
    {"label": {"en": "⬅ Back", "hi": "⬅ वापस"}, "action": "menu:main"},
]

MINTING_MENU = [
    {"label": {"en": "⚙️ What is Minting?", "hi": "⚙️ मिंटिंग क्या है?"}, "action": "q:minting_info"},
    {"label": {"en": "⏱ When Minting Happens?", "hi": "⏱ मिंटिंग कब होती है?"}, "action": "q:minting_time"},
    {"label": {"en": "📍 Minted Token Location", "hi": "📍 टोकन कहाँ मिलता है?"}, "action": "q:minting_location"},
    {"label": {"en": "⬅ Back", "hi": "⬅ वापस"}, "action": "menu:main"},
]

LP_MENU = [
    {"label": {"en": "💧 What is Liquidity Pool?", "hi": "💧 लिक्विडिटी पूल क्या है?"}, "action": "q:lp_info"},
    {"label": {"en": "🔗 LP Pair", "hi": "🔗 एलपी पेयर"}, "action": "q:lp_pair"},
    {"label": {"en": "⭐ Benefits of LP", "hi": "⭐ लाभ"}, "action": "q:lp_benefits"},
    {"label": {"en": "💰 LP Rewards", "hi": "💰 रिवार्ड"}, "action": "q:lp_rewards"},
    {"label": {"en": "⬅ Back", "hi": "⬅ वापस"}, "action": "menu:others"},
]

STAKING_MENU = [
    {"label": {"en": "🔐 What is Staking?", "hi": "🔐 स्टेकिंग क्या है?"}, "action": "q:staking_info"},
    {"label": {"en": "⚙ How Staking Works", "hi": "⚙ कैसे काम करता है"}, "action": "q:staking_work"},
    {"label": {"en": "💰 Rewards from Staking", "hi": "💰 रिवार्ड"}, "action": "q:staking_rewards"},
    {"label": {"en": "⬅ Back", "hi": "⬅ वापस"}, "action": "menu:others"},
]

WITHDRAW_MENU = [
    {"label": {"en": "💸 Can I withdraw anytime?", "hi": "💸 क्या कभी भी निकाल सकते हैं?"}, "action": "q:withdraw_anytime"},
    {"label": {"en": "💰 Withdrawal currency", "hi": "💰 निकासी करेंसी"}, "action": "q:withdraw_currency"},
    {"label": {"en": "🔥 Token burning mechanism", "hi": "🔥 टोकन बर्न सिस्टम"}, "action": "q:withdraw_burn"},
    {"label": {"en": "⬅ Back", "hi": "⬅ वापस"}, "action": "menu:others"},
]

AIRDROP_MENU = [
    {"label": {"en": "🎁 Airdrop eligibility", "hi": "🎁 पात्रता"}, "action": "q:airdrop_eligibility"},
    {"label": {"en": "💰 Airdrop reward", "hi": "💰 इनाम"}, "action": "q:airdrop_reward"},
    {"label": {"en": "📋 Conditions", "hi": "📋 शर्तें"}, "action": "q:airdrop_conditions"},
    {"label": {"en": "⬅ Back", "hi": "⬅ वापस"}, "action": "menu:others"},
]

AFFILIATE_MENU = [
    {"label": {"en": "🤝 Affiliate Program", "hi": "🤝 एफिलिएट प्रोग्राम"}, "action": "q:affiliate_info"},
    {"label": {"en": "👥 Team Business", "hi": "👥 टीम बिजनेस"}, "action": "q:affiliate_team"},
    {"label": {"en": "📈 Importance", "hi": "📈 महत्व"}, "action": "q:affiliate_importance"},
    {"label": {"en": "⬅ Back", "hi": "⬅ वापस"}, "action": "menu:others"},
]

RANKS_MENU = [
    {"label": {"en": "🏅 Rank Structure", "hi": "🏅 रैंक संरचना"}, "action": "q:rank_structure"},
    {"label": {"en": "🎖 Club Rewards", "hi": "🎖 क्लब रिवार्ड"}, "action": "q:club_rewards"},
    {"label": {"en": "📊 Requirements", "hi": "📊 आवश्यकताएँ"}, "action": "q:rank_requirements"},
    {"label": {"en": "⬅ Back", "hi": "⬅ वापस"}, "action": "menu:others"},
]

TRIPLE_MENU = [
    {"label": {"en": "💎 Triple Income System", "hi": "💎 ट्रिपल इनकम सिस्टम"}, "action": "q:triple_info"},
    {"label": {"en": "📉 After Limit", "hi": "📉 लिमिट के बाद"}, "action": "q:triple_limit"},
    {"label": {"en": "⬅ Back", "hi": "⬅ वापस"}, "action": "menu:others"},
]

LANGUAGE_MENU = {
    "🇬🇧 English": "lang:en",
    "🇮🇳 Hindi": "lang:hi",
    "🇮🇳 Marathi": "lang:mr",
    "🇮🇳 Bengali": "lang:bn",
}

HARDCODED_ANSWERS = {
    "wallet_info": "A FINUX wallet is a digital wallet where your *FNX tokens and rewards* are stored.\nIt is *automatically generated* when you register in the system.",

"wallet_create": "1️⃣ Download the wallet from the official website:\nhttps://finux.online\n2️⃣ Your wallet will be generated automatically.\n⚠️ Secure your *private key / seed phrase*.\n3️⃣ Sign up on DEX to start using your wallet.",

"wallet_security": "FINUX wallets operate in a secure blockchain environment.\nHowever, users must protect their *private key or seed phrase*.\n⚠️ If you lose it, the company *cannot recover your funds*.",

"wallet_private": "Your private key or seed phrase is a *secret code* that gives access to your wallet.\n⚠️ Never share it with anyone.\nAnyone with this key can *control your funds*.",
    
"deposit_min": "The minimum deposit is *$20*.",

"deposit_plans": "You can deposit:\n• $20\n• $50\n• $100\n• $200\n• Multiples of $100",

"deposit_structure": "Your deposit is split into:\n• 30% MSTC\n• 70% USDC (Polygon Network)",

"deposit_blockchain": "The system uses *MEP-20 blockchain contract*.",    
    
"minting_info": "Minting means creating a new FNX token in the system.",

"minting_time": "After your deposit transaction is completed.",

"minting_location": "The system automatically credits the minted FNX token to your wallet.",    
    
"lp_info": "A Liquidity Pool is where users provide tokens to help trading happen smoothly.",

"lp_pair": "FNX + USDC pair is used.",

"lp_benefits": "Stable trading\n• Daily passive income\n• High rewards\n• Community growth\n• Strong ecosystem support",

"lp_rewards": "You can earn daily rewards up to *5% MPY (Monthly Percentage Yield)*.\nThese rewards are generated from the system's trading and ecosystem activity.",
    
"staking_info": "It means locking FNX tokens in the system to earn rewards.",

"staking_work": "The staking process is very simple:\n• Deposit funds into the platform\n• FNX tokens are minted and credited to your wallet\n• Stake your FNX tokens in the Self-Staking section\n• The system generates daily rewards automatically\n• You can withdraw rewards anytime",

"staking_rewards": "Up to *2% MPY (Monthly Percentage Yield)* daily reward.",    
    
"withdraw_anytime": "Yes, FNX rewards can be withdrawn anytime.",

"withdraw_currency": "You will receive *USDC* in your wallet instantly.",

"withdraw_burn": "When you withdraw FNX:\n• 50% FNX is burned\n• 50% FNX goes back to supply.\nThis helps control token supply.",    
    
"airdrop_eligibility": "Yes, you must have at least *5 direct paid referrals*.",

"airdrop_reward": "You receive *50 FNX tokens*.",

"airdrop_conditions": "• Wallet must be registered\n• User must be verified\n• Duplicate referrals are not counted",   
    
"affiliate_info": "It is a referral program where you earn rewards by building a team.",

"affiliate_team": "The total deposits made by your team.",

"affiliate_importance": "It helps grow the community and increases earnings.",    
    
"rank_structure": "• Rank 1 — Origin 10%\n• Rank 2 — Life Changer 16%\n• Rank 3 — Advisor 20%\n• Rank 4 — Visionary 23%\n• Rank 5 — Creator 25%",

"club_rewards": "• Rank 1 (Origin) — 10%\n• Rank 2 (Life Changer) — 16% (3% CTO club share)\n• Rank 3 (Advisor) — 20% (2.5% CTO club share)\n• Rank 4 (Visionary) — 23% (2% CTO club share)\n• Rank 5 (Creator) — 25% (1.5% CTO club share)",

"rank_requirements": "• Rank 1 (Origin)\n  • Self activation\n• Rank 2 (Life Changer)\n  • $1000 team business\n  • 10 active origins\n  • Minimum $30 LP\n• Rank 3 (Advisor)\n  • $5000 team business\n  • 2 active life changers\n  • Minimum $100 LP\n• Rank 4 (Visionary)\n  • $25,000 team business\n  • 2 active advisors\n  • Minimum $300 LP\n• Rank 5 (Creator)\n  • $100,000 team business\n  • 2 active visionaries\n  • Minimum $1000 LP",    
    
"triple_info": "Users can earn from three sources:\n• Performance income — up to 3x\n• Liquidity pool reward — up to 3x\n• FNX staking — up to 2x",

"triple_limit": "After *3x performance income*, you must *retop-up* to continue earning.",    
    
"terms_conditions": "*General T&C*\n• Anyone can join the program\n• Rewards based on company policy\n• Company may update program anytime\n\n*Airdrop T&C*\n• Wallet registration + verification\n• Limited period\n• Duplicate referrals not counted\n\n*Additional T&C*\n• LP 50% counts in team business\n• Performance income limit: 3X\n• Retop-up required after limit\n• Retop-up gives 50% FNX",   
    
"risk_disclaimer": "• Crypto investments carry risk\n• Earnings are not guaranteed\n• Users must secure their wallets\n• Company is not responsible for lost private keys",    
   
 }


TRANSLATION_CACHE = {}

# ===================== UI HELPERS =====================


def header_buttons():
    return [
        [{"text": "🌐 Change Language", "callback_data": "change_lang"}],
        [{"text": " Open App", "url": "https://finux-chatbot-production.up.railway.app"}],
        [
            {"text": "Channel", "url": "https://t.me/Finuxofficiallive"},
            {"text": " Website", "url": "https://finux.online/"},
        ],
    ]

def get_full_menu(menu_name):
    if menu_name == "main":
        return MAIN_MENU
    elif menu_name == "wallet":
        return WALLET_MENU
    elif menu_name == "deposit":
        return DEPOSIT_MENU
    elif menu_name == "minting":
        return MINTING_MENU
    elif menu_name == "others":
        return OTHERS_MENU
    elif menu_name == "lp":
        return LP_MENU
    elif menu_name == "staking":
        return STAKING_MENU
    elif menu_name == "withdraw":
        return WITHDRAW_MENU
    elif menu_name == "airdrop":
        return AIRDROP_MENU
    elif menu_name == "affiliate":
        return AFFILIATE_MENU
    elif menu_name == "ranks":
        return RANKS_MENU
    elif menu_name == "triple_income":
        return TRIPLE_MENU
    return []

def get_user_language(user_id):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT language FROM users WHERE username=%s",
        (user_id,)
    )

    row = cur.fetchone()

    cur.close()
    conn.close()

    return row[0] if row else "en"

def get_full_answer(key, user_id=None):
    lang = "en"

    if user_id:
        lang = get_user_language(user_id)

    # DB answer
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT answer FROM custom_answers WHERE key=%s",
        (key,)
    )

    row = cur.fetchone()

    cur.close()
    conn.close()

    if row:
        return row[0]

    answer = HARDCODED_ANSWERS.get(key)

    if isinstance(answer, dict):
        return answer.get(lang, answer.get("en"))

    return answer


def build_menu(menu_key, user_id=None):

    keyboard = header_buttons()
    menu_items = get_full_menu(menu_key)

    lang = "en"
    if user_id:
        lang = get_user_language(user_id)

    row = []

    for item in menu_items:

        label_dict = item["label"]
        action = item["action"]

        # 🔥 pick correct language
        text = label_dict.get(lang, label_dict.get("en"))

        row.append({
            "text": text,
            "callback_data": action
        })

        if len(row) == 2:
            keyboard.append(row)
            row = []

    if row:
        keyboard.append(row)

    return {"inline_keyboard": keyboard}

def is_admin(username):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT role FROM users WHERE username=%s", (username,))
    row = cur.fetchone()

    cur.close()
    conn.close()

    return row and row[0] == "admin"


# ===================== FASTAPI =====================

app = FastAPI()

@app.on_event("startup")
def startup():
  init_db()

from app.auth import router as auth_router

app.include_router(auth_router)

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(DATA_DIR, "ui.html"))

@app.get("/login")
async def login_page():
    return FileResponse(os.path.join(DATA_DIR, "login.html"))


from fastapi import Header, HTTPException
from jose import jwt

SECRET_KEY = "finux-secret-key"
ALGORITHM = "HS256"

from jose import jwt, JWTError

@app.post("/chat")
async def chat_api(payload: ChatRequest, request: Request):

    auth_header = request.headers.get("Authorization")

    if not auth_header:
        return {"response": "Please login first."}

    try:
        token = auth_header.split(" ")[1]
        data = jwt.decode(token, "finux-secret-key", algorithms=["HS256"])
        user = data.get("sub")
        print("USER:", user)
    except JWTError:
        return {"response": "Invalid or expired token. Please login again."}

    # ✅ NEW LINE
    session_id = payload.session_id
    print("SESSION:", session_id)

    question = payload.message.strip()
    answer = generate_answer(question)
    lang = get_user_language(user)
    answer = translate(answer, lang)

    # ✅ UPDATED save_chat
    try:
        save_chat(
            "web",
            user,
            session_id,   # 🔥 IMPORTANT CHANGE
            question,
            answer
        )
    except Exception as e:
        logging.error(f"DB save error: {e}")

    return {"response": answer}

@app.get("/sessions")
def get_sessions(request: Request):

    auth_header = request.headers.get("Authorization")

    if not auth_header:
        return []

    token = auth_header.split(" ")[1]
    data = jwt.decode(token, "finux-secret-key", algorithms=["HS256"])
    user = data.get("sub")

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    SELECT session_id, MIN(question), MIN(created_at)
    FROM chats
    WHERE user_id=%s
    GROUP BY session_id
    ORDER BY MIN(created_at) DESC
""", (user,))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
    {"session_id": r[0], "title": r[1][:40] if r[1] else "New Chat"}
    for r in rows if r[0]
]

@app.get("/session/{session_id}")
def get_session_chat(session_id: str, request: Request):

    token = request.headers.get("Authorization").split(" ")[1]
    data = jwt.decode(token, "finux-secret-key", algorithms=["HS256"])
    user = data.get("sub")

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT question, answer
        FROM chats
        WHERE user_id=%s AND session_id=%s
        ORDER BY created_at
    """, (user, session_id))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [{"q": r[0], "a": r[1]} for r in rows]

@app.get("/me")
def get_me(request: Request):

    from jose import JWTError, ExpiredSignatureError

    auth_header = request.headers.get("Authorization")

    if not auth_header:
        return {"error": "Not logged in"}

    try:
        token = auth_header.split(" ")[1]
        data = jwt.decode(token, "finux-secret-key", algorithms=["HS256"])
        username = data.get("sub")

    except ExpiredSignatureError:
        return {"error": "Session expired"}
    except JWTError:
        return {"error": "Invalid token"}

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT role FROM users WHERE username=%s",
        (username,)
    )

    row = cur.fetchone()

    cur.close()
    conn.close()

    return {
        "username": username,
        "role": row[0] if row else "user"
    }

@app.post("/admin/add-full")
async def add_full(payload: dict, request: Request):

    token = request.headers.get("Authorization").split(" ")[1]
    data = jwt.decode(token, "finux-secret-key", algorithms=["HS256"])
    username = data.get("sub")

    if not is_admin(username):
        return {"error": "Not authorized"}

    conn = get_conn()
    cur = conn.cursor()

    # 1️⃣ Insert answer
    cur.execute(
        "INSERT INTO custom_answers (key, answer) VALUES (%s,%s)",
        (payload["key"], payload["answer"])
    )

    # 2️⃣ Insert menu
    cur.execute("""
        INSERT INTO custom_menus (label, type, target, parent)
        VALUES (%s,%s,%s,%s)
    """, (
        payload["label"],
        "question",
        f"q:{payload['key']}",
        payload["parent"]
    ))

    conn.commit()
    cur.close()
    conn.close()

    return {"message": "✅ Menu added successfully"}


@app.post("/admin/delete-menu")
async def delete_menu(payload: dict, request: Request):

    token = request.headers.get("Authorization").split(" ")[1]
    data = jwt.decode(token, "finux-secret-key", algorithms=["HS256"])
    username = data.get("sub")

    if not is_admin(username):
        return {"error": "Not authorized"}

    conn = get_conn()
    cur = conn.cursor()

    # delete menu
    cur.execute(
        "DELETE FROM custom_menus WHERE label=%s",
        (payload["label"],)
    )

    # delete answer
    cur.execute(
        "DELETE FROM custom_answers WHERE key=%s",
        (payload["key"],)
    )

    conn.commit()
    cur.close()
    conn.close()

    return {"message": "🗑 Menu deleted successfully"}

@app.get("/admin/menus")
def get_all_menus(request: Request):

    token = request.headers.get("Authorization").split(" ")[1]
    data = jwt.decode(token, "finux-secret-key", algorithms=["HS256"])
    username = data.get("sub")

    if not is_admin(username):
        return []

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT label, target, parent FROM custom_menus
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "label": r[0],
            "key": r[1].replace("q:", ""),
            "parent": r[2]
        }
        for r in rows
    ]


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

            # 🌐 LANGUAGE SELECT
            if payload.startswith("lang:"):
                lang = payload.split(":")[1]

                conn = get_conn()
                cur = conn.cursor()

                # ensure user exists
                cur.execute(
                    "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING",
                    (str(chat_id), "telegram_user", "user")
                )

                # update language
                cur.execute(
                    "UPDATE users SET language=%s WHERE username=%s",
                    (lang, str(chat_id))
                )

                conn.commit()
                cur.close()
                conn.close()

                await client.post(
                    f"{TELEGRAM_API}/editMessageText",
                    json={
                        "chat_id": chat_id,
                        "message_id": cq["message"]["message_id"],
                        "text": "✅ Language selected!\n\nPlease choose an option:",
                        "reply_markup": build_menu("main", str(chat_id)),
                    },
                )

                return {"ok": True}

            # 🌐 CHANGE LANGUAGE BUTTON
            if payload == "change_lang":
                await client.post(
                    f"{TELEGRAM_API}/editMessageText",
                    json={
                        "chat_id": chat_id,
                        "message_id": cq["message"]["message_id"],
                        "text": "🌐 *Please choose your language:*",
                        "parse_mode": "Markdown",
                        "reply_markup": {
                            "inline_keyboard": [
                                [{"text": k, "callback_data": v}] for k, v in LANGUAGE_MENU.items()
                            ]
                        },
                    },
                )
                return {"ok": True}

            # 📂 MENU NAVIGATION
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
                        "reply_markup": build_menu(menu_key, str(chat_id)),
                    },
                )

                return {"ok": True}

            # ❓ QUESTION HANDLER
            if payload.startswith("q:"):
                key = payload.replace("q:", "")

                # 1️⃣ Hardcoded
                answer = get_full_answer(key, str(chat_id))

                # 2️⃣ Document search
                if not answer:
                    topic = key.replace("_", " ")
                    answer = semantic_search(topic)

                # 3️⃣ Gemini fallback
                if not answer:
                    topic = key.replace("_", " ")
                    answer = generate_answer(topic)

                # 4️⃣ Final fallback
                if not answer:
                    answer = "No information available."

                # 🌐 Translate
                lang = get_user_language(str(chat_id))
                answer = translate(answer, lang)

                message_id = cq["message"]["message_id"]

                # determine menu
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
                elif key in ["rank_structure", "club_rewards", "rank_requirements"]:
                    menu_to_show = "ranks"
                elif key.startswith("terms") or key.startswith("risk"):
                    menu_to_show = "others"
                elif key.startswith("triple"):
                    menu_to_show = "triple_income"

                await client.post(
                    f"{TELEGRAM_API}/editMessageText",
                    json={
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "text": answer,
                        "parse_mode": "Markdown",
                        "reply_markup": build_menu(menu_to_show, str(chat_id)),
                    },
                )

                # save chat
                try:
                    save_chat("telegram", str(chat_id), "", key, answer)
                except Exception as e:
                    logging.error(f"DB save error (callback): {e}")

                return {"ok": True}

        # ================= NORMAL MESSAGE =================
        message = data.get("message")
        if not message:
            return {"ok": True}

        chat_id = message["chat"]["id"]
        text = message.get("text", "").strip()

        # /start
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
                    "text": "🌐 *Please choose your language:*",
                    "parse_mode": "Markdown",
                    "reply_markup": {
                        "inline_keyboard": [
                            [{"text": k, "callback_data": v}] for k, v in LANGUAGE_MENU.items()
                        ]
                    },
                },
            )

            return {"ok": True}

        # user message
        if text:
            answer = generate_answer(text)

            lang = get_user_language(str(chat_id))
            answer = translate(answer, lang)

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