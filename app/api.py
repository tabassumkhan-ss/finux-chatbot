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

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "vi": "Vietnamese",
    "tl": "Filipino",
    "ru": "Russian",
    "th": "Thai"
}

def translate(text, lang):
    if not text or lang == "en":
        return text

    language_name = LANG_NAMES.get(lang, "English")

    try:
        prompt = f"""
Translate the following text into {language_name}.
Keep formatting, emojis, and line breaks same.

{text}
"""

        response = client.models.generate_content(
            model="models/gemini-flash-latest",
            contents=prompt
        )

        if response.text:
            return response.text.strip()

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
            "mr": "💼 वॉलेट",
            "bn": "💼 ওয়ালেট",
            "vi": "💼 Ví",
            "tl": "💼 Wallet",
            "ru": "💼 Кошелек",
            "th": "💼 กระเป๋าเงิน"
        },
        "action": "menu:wallet"
    },
    {
        "label": {
            "en": "💰 Deposit",
            "hi": "💰 जमा",
            "mr": "💰 जमा",
            "bn": "💰 জমা",
            "vi": "💰 Nạp tiền",
            "tl": "💰 Deposito",
            "ru": "💰 Депозит",
            "th": "💰 ฝากเงิน"
        },
        "action": "menu:deposit"
    },
    {
        "label": {
            "en": "🪙 Minting",
            "hi": "🪙 मिंटिंग",
            "mr": "🪙 मिंटिंग",
            "bn": "🪙 মিন্টিং",
            "vi": "🪙 Đúc token",
            "tl": "🪙 Minting",
            "ru": "🪙 Минтинг",
            "th": "🪙 การสร้างเหรียญ"
        },
        "action": "menu:minting"
    },
    {
        "label": {
            "en": "📦 Others",
            "hi": "📦 अन्य",
            "mr": "📦 इतर",
            "bn": "📦 অন্যান্য",
            "vi": "📦 Khác",
            "tl": "📦 Iba pa",
            "ru": "📦 Другое",
            "th": "📦 อื่นๆ"
        },
        "action": "menu:others"
    }
]

OTHERS_MENU = [
    {
        "label": {
            "en": "💧 Liquidity Pool",
            "hi": "💧 लिक्विडिटी पूल",
            "mr": "💧 लिक्विडिटी पूल",
            "bn": "💧 লিকুইডিটি পুল",
            "vi": "💧 Thanh khoản",
            "tl": "💧 Liquidity Pool",
            "ru": "💧 Пул ликвидности",
            "th": "💧 สภาพคล่อง"
        },
        "action": "menu:lp"
    },
    {
        "label": {
            "en": "🔐 Staking",
            "hi": "🔐 स्टेकिंग",
            "mr": "🔐 स्टेकिंग",
            "bn": "🔐 স্টেকিং",
            "vi": "🔐 Staking",
            "tl": "🔐 Staking",
            "ru": "🔐 Стейкинг",
            "th": "🔐 การ Stake"
        },
        "action": "menu:staking"
    },
    {
        "label": {
            "en": "💸 Withdraw",
            "hi": "💸 निकासी",
            "mr": "💸 पैसे काढा",
            "bn": "💸 উত্তোলন",
            "vi": "💸 Rút tiền",
            "tl": "💸 Withdraw",
            "ru": "💸 Вывод",
            "th": "💸 ถอนเงิน"
        },
        "action": "menu:withdraw"
    },
    {
        "label": {
            "en": "⬅ Back",
            "hi": "⬅ वापस",
            "mr": "⬅ मागे",
            "bn": "⬅ পিছনে",
            "vi": "⬅ Quay lại",
            "tl": "⬅ Bumalik",
            "ru": "⬅ Назад",
            "th": "⬅ กลับ"
        },
        "action": "menu:main"
    }
]

WALLET_MENU = [
    {
        "label": {
            "en": "What is Wallet?",
            "hi": "वॉलेट क्या है?",
            "mr": "वॉलेट म्हणजे काय?",
            "bn": "ওয়ালেট কি?",
            "vi": "Ví là gì?",
            "tl": "Ano ang Wallet?",
            "ru": "Что такое кошелек?",
            "th": "กระเป๋าเงินคืออะไร?"
        },
        "action": "q:wallet_info"
    },
    {
        "label": {
            "en": "Create Wallet",
            "hi": "वॉलेट बनाएं",
            "mr": "वॉलेट तयार करा",
            "bn": "ওয়ালেট তৈরি করুন",
            "vi": "Tạo ví",
            "tl": "Gumawa ng Wallet",
            "ru": "Создать кошелек",
            "th": "สร้างกระเป๋าเงิน"
        },
        "action": "q:wallet_create"
    },
    {
        "label": {
            "en": "Wallet Security",
            "hi": "वॉलेट सुरक्षा",
            "mr": "वॉलेट सुरक्षा",
            "bn": "ওয়ালেট নিরাপত্তা",
            "vi": "Bảo mật ví",
            "tl": "Seguridad ng Wallet",
            "ru": "Безопасность кошелька",
            "th": "ความปลอดภัยของกระเป๋าเงิน"
        },
        "action": "q:wallet_security"
    },
    {
        "label": {
            "en": "Private Key",
            "hi": "प्राइवेट की",
            "mr": "प्रायव्हेट की",
            "bn": "প্রাইভেট কি",
            "vi": "Khóa riêng",
            "tl": "Private Key",
            "ru": "Приватный ключ",
            "th": "คีย์ส่วนตัว"
        },
        "action": "q:wallet_private"
    },
    {
        "label": {
            "en": "⬅ Back",
            "hi": "⬅ वापस",
            "mr": "⬅ मागे",
            "bn": "⬅ পিছনে",
            "vi": "⬅ Quay lại",
            "tl": "⬅ Bumalik",
            "ru": "⬅ Назад",
            "th": "⬅ กลับ"
        },
        "action": "menu:main"
    }
]

DEPOSIT_MENU = [
    {
        "label": {
            "en": "💰 Minimum Deposit",
            "hi": "💰 न्यूनतम जमा",
            "mr": "💰 किमान जमा",
            "bn": "💰 ন্যূনতম জমা",
            "vi": "💰 Nạp tối thiểu",
            "tl": "💰 Minimum na Deposito",
            "ru": "💰 Минимальный депозит",
            "th": "💰 ฝากขั้นต่ำ"
        },
        "action": "q:deposit_min"
    },
    {
        "label": {
            "en": "📊 Deposit Plans",
            "hi": "📊 जमा योजनाएं",
            "mr": "📊 जमा योजना",
            "bn": "📊 জমার পরিকল্পনা",
            "vi": "📊 Gói nạp tiền",
            "tl": "📊 Mga Plano ng Deposito",
            "ru": "📊 Планы депозита",
            "th": "📊 แผนการฝาก"
        },
        "action": "q:deposit_plans"
    },
    {
        "label": {
            "en": "📦 Deposit Structure",
            "hi": "📦 जमा संरचना",
            "mr": "📦 जमा संरचना",
            "bn": "📦 জমার গঠন",
            "vi": "📦 Cấu trúc nạp",
            "tl": "📦 Istruktura ng Deposito",
            "ru": "📦 Структура депозита",
            "th": "📦 โครงสร้างการฝาก"
        },
        "action": "q:deposit_structure"
    },
    {
        "label": {
            "en": "⛓ Blockchain",
            "hi": "⛓ ब्लॉकचेन",
            "mr": "⛓ ब्लॉकचेन",
            "bn": "⛓ ব্লকচেইন",
            "vi": "⛓ Blockchain",
            "tl": "⛓ Blockchain",
            "ru": "⛓ Блокчейн",
            "th": "⛓ บล็อกเชน"
        },
        "action": "q:deposit_blockchain"
    },
    {
        "label": {
            "en": "⬅ Back",
            "hi": "⬅ वापस",
            "mr": "⬅ मागे",
            "bn": "⬅ পিছনে",
            "vi": "⬅ Quay lại",
            "tl": "⬅ Bumalik",
            "ru": "⬅ Назад",
            "th": "⬅ กลับ"
        },
        "action": "menu:main"
    }
]

MINTING_MENU = [
    {
        "label": {
            "en": "⚙️ What is Minting?",
            "hi": "⚙️ मिंटिंग क्या है?",
            "mr": "⚙️ मिंटिंग म्हणजे काय?",
            "bn": "⚙️ মিন্টিং কী?",
            "vi": "⚙️ Minting là gì?",
            "tl": "⚙️ Ano ang Minting?",
            "ru": "⚙️ Что такое минтинг?",
            "th": "⚙️ การ Mint คืออะไร?"
        },
        "action": "q:minting_info"
    },
    {
        "label": {
            "en": "⏱ When Minting Happens?",
            "hi": "⏱ मिंटिंग कब होती है?",
            "mr": "⏱ मिंटिंग कधी होते?",
            "bn": "⏱ মিন্টিং কখন হয়?",
            "vi": "⏱ Khi nào mint?",
            "tl": "⏱ Kailan nangyayari ang Minting?",
            "ru": "⏱ Когда происходит минтинг?",
            "th": "⏱ การ Mint เกิดขึ้นเมื่อไร?"
        },
        "action": "q:minting_time"
    },
    {
        "label": {
            "en": "📍 Token Location",
            "hi": "📍 टोकन कहाँ मिलता है?",
            "mr": "📍 टोकन कुठे मिळतो?",
            "bn": "📍 টোকেন কোথায় পাওয়া যায়?",
            "vi": "📍 Token ở đâu?",
            "tl": "📍 Saan napupunta ang Token?",
            "ru": "📍 Где находятся токены?",
            "th": "📍 โทเค็นอยู่ที่ไหน?"
        },
        "action": "q:minting_location"
    },
    {
        "label": {
            "en": "⬅ Back",
            "hi": "⬅ वापस",
            "mr": "⬅ मागे",
            "bn": "⬅ পিছনে",
            "vi": "⬅ Quay lại",
            "tl": "⬅ Bumalik",
            "ru": "⬅ Назад",
            "th": "⬅ กลับ"
        },
        "action": "menu:main"
    }
]

LP_MENU = [
    {
        "label": {
            "en": "💧 What is LP?",
            "hi": "💧 लिक्विडिटी पूल क्या है?",
            "mr": "💧 लिक्विडिटी पूल म्हणजे काय?",
            "bn": "💧 লিকুইডিটি পুল কী?",
            "vi": "💧 LP là gì?",
            "tl": "💧 Ano ang LP?",
            "ru": "💧 Что такое LP?",
            "th": "💧 LP คืออะไร?"
        },
        "action": "q:lp_info"
    },
    {
        "label": {
            "en": "🔗 LP Pair",
            "hi": "🔗 एलपी पेयर",
            "mr": "🔗 एलपी पेअर",
            "bn": "🔗 এলপি পেয়ার",
            "vi": "🔗 Cặp LP",
            "tl": "🔗 LP Pair",
            "ru": "🔗 Пара LP",
            "th": "🔗 คู่ LP"
        },
        "action": "q:lp_pair"
    },
    {
        "label": {
            "en": "⭐ Benefits",
            "hi": "⭐ लाभ",
            "mr": "⭐ फायदे",
            "bn": "⭐ সুবিধা",
            "vi": "⭐ Lợi ích",
            "tl": "⭐ Benepisyo",
            "ru": "⭐ Преимущества",
            "th": "⭐ ประโยชน์"
        },
        "action": "q:lp_benefits"
    },
    {
        "label": {
            "en": "💰 Rewards",
            "hi": "💰 रिवार्ड",
            "mr": "💰 रिवॉर्ड",
            "bn": "💰 পুরস্কার",
            "vi": "💰 Phần thưởng",
            "tl": "💰 Gantimpala",
            "ru": "💰 Награды",
            "th": "💰 รางวัล"
        },
        "action": "q:lp_rewards"
    },
    {
        "label": {
            "en": "⬅ Back",
            "hi": "⬅ वापस",
            "mr": "⬅ मागे",
            "bn": "⬅ পিছনে",
            "vi": "⬅ Quay lại",
            "tl": "⬅ Bumalik",
            "ru": "⬅ Назад",
            "th": "⬅ กลับ"
        },
        "action": "menu:others"
    }
]

STAKING_MENU = [
    {
        "label": {
            "en": "🔐 What is Staking?",
            "hi": "🔐 स्टेकिंग क्या है?",
            "mr": "🔐 स्टेकिंग म्हणजे काय?",
            "bn": "🔐 স্টেকিং কী?",
            "vi": "🔐 Staking là gì?",
            "tl": "🔐 Ano ang Staking?",
            "ru": "🔐 Что такое стейкинг?",
            "th": "🔐 การ Stake คืออะไร?"
        },
        "action": "q:staking_info"
    },
    {
        "label": {
            "en": "⚙ How it Works",
            "hi": "⚙ कैसे काम करता है",
            "mr": "⚙ कसे कार्य करते",
            "bn": "⚙ কিভাবে কাজ করে",
            "vi": "⚙ Cách hoạt động",
            "tl": "⚙ Paano gumagana",
            "ru": "⚙ Как это работает",
            "th": "⚙ วิธีการทำงาน"
        },
        "action": "q:staking_work"
    },
    {
        "label": {
            "en": "💰 Rewards",
            "hi": "💰 रिवार्ड",
            "mr": "💰 रिवॉर्ड",
            "bn": "💰 পুরস্কার",
            "vi": "💰 Phần thưởng",
            "tl": "💰 Gantimpala",
            "ru": "💰 Награды",
            "th": "💰 รางวัล"
        },
        "action": "q:staking_rewards"
    },
    {
        "label": {
            "en": "⬅ Back",
            "hi": "⬅ वापस",
            "mr": "⬅ मागे",
            "bn": "⬅ পিছনে",
            "vi": "⬅ Quay lại",
            "tl": "⬅ Bumalik",
            "ru": "⬅ Назад",
            "th": "⬅ กลับ"
        },
        "action": "menu:others"
    }
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
    "🇻🇳 Vietnamese": "lang:vi",
    "🇵🇭 Filipino": "lang:tl",
    "🇷🇺 Russian": "lang:ru",
    "🇹🇭 Thai": "lang:th"
}

def build_language_menu():
    items = list(LANGUAGE_MENU.items())
    keyboard = []
    row = []

    for label, value in items:
        row.append({"text": label, "callback_data": value})

        if len(row) == 2:
            keyboard.append(row)
            row = []

    if row:  # if odd number
        keyboard.append(row)

    return {"inline_keyboard": keyboard}

HARDCODED_ANSWERS = {
   "wallet_info": {
    "en": "A FINUX wallet is a digital wallet where your FNX tokens and rewards are stored.",
    "hi": "FINUX वॉलेट एक डिजिटल वॉलेट है जहाँ आपके FNX टोकन और रिवॉर्ड सुरक्षित रहते हैं।",
    "mr": "FINUX वॉलेट हे एक डिजिटल वॉलेट आहे जिथे तुमचे FNX टोकन आणि रिवॉर्ड साठवले जातात.",
    "bn": "FINUX ওয়ালেট একটি ডিজিটাল ওয়ালেট যেখানে আপনার FNX টোকেন সংরক্ষিত হয়।",
    "vi": "Ví FINUX là ví kỹ thuật số nơi lưu trữ FNX và phần thưởng của bạn.",
    "tl": "Ang FINUX wallet ay isang digital wallet kung saan nakaimbak ang FNX tokens at rewards.",
    "ru": "FINUX кошелек — это цифровой кошелек для хранения токенов FNX и вознаграждений.",
    "th": "กระเป๋า FINUX เป็นกระเป๋าดิจิทัลสำหรับเก็บโทเค็น FNX และรางวัลของคุณ"
},

"wallet_create": {
    "en": "Download wallet from official website. Secure your private key.",
    "hi": "ऑफिशियल वेबसाइट से वॉलेट डाउनलोड करें और अपनी प्राइवेट की सुरक्षित रखें।",
    "mr": "अधिकृत वेबसाइटवरून वॉलेट डाउनलोड करा आणि तुमची प्रायव्हेट की सुरक्षित ठेवा.",
    "bn": "অফিসিয়াল ওয়েবসাইট থেকে ওয়ালেট ডাউনলোড করুন এবং প্রাইভেট কি সুরক্ষিত রাখুন।",
    "vi": "Tải ví từ website chính thức và bảo mật khóa riêng của bạn.",
    "tl": "I-download ang wallet mula sa opisyal na website at panatilihing ligtas ang private key.",
    "ru": "Скачайте кошелек с официального сайта и защитите приватный ключ.",
    "th": "ดาวน์โหลดกระเป๋าจากเว็บไซต์ทางการและเก็บคีย์ส่วนตัวให้ปลอดภัย"
},

"wallet_security": {
    "en": "Keep your private key safe. If lost, funds cannot be recovered.",
    "hi": "अपनी प्राइवेट की सुरक्षित रखें, खोने पर फंड वापस नहीं मिलेंगे।",
    "mr": "तुमची प्रायव्हेट की सुरक्षित ठेवा, हरवल्यास निधी परत मिळणार नाही.",
    "bn": "আপনার প্রাইভেট কি নিরাপদ রাখুন, হারালে ফান্ড ফিরে পাবেন না।"
},

"wallet_private": {
    "en": "Private key gives full control of your wallet. Never share it.",
    "hi": "प्राइवेट की आपके वॉलेट का पूरा नियंत्रण देती है, इसे कभी शेयर न करें।",
    "mr": "प्रायव्हेट की तुमच्या वॉलेटवर पूर्ण नियंत्रण देते, कधीही शेअर करू नका.",
    "bn": "প্রাইভেট কি আপনার ওয়ালেটের সম্পূর্ণ নিয়ন্ত্রণ দেয়, কখনও শেয়ার করবেন না।"
},
    
"deposit_min": {
    "en": "Minimum deposit is $20.",
    "hi": "न्यूनतम जमा $20 है।",
    "mr": "किमान जमा $20 आहे.",
    "bn": "ন্যূনতম জমা $20।",
    "vi": "Số tiền nạp tối thiểu là $20.",
    "tl": "Ang minimum na deposito ay $20.",
    "ru": "Минимальный депозит составляет $20.",
    "th": "ยอดฝากขั้นต่ำคือ $20"
},

"deposit_plans": {
    "en": "You can deposit $20, $50, $100, $200 or multiples of $100.",
    "hi": "$20, $50, $100, $200 या $100 के गुणक जमा कर सकते हैं।",
    "mr": "$20, $50, $100, $200 किंवा $100 च्या पटीत जमा करू शकता.",
    "bn": "$20, $50, $100, $200 বা $100 এর গুণিতক জমা করা যায়।",
    "vi": "Bạn có thể nạp $20, $50, $100, $200 hoặc bội số của $100.",
    "tl": "Maaari kang magdeposito ng $20, $50, $100, $200 o multiples ng $100.",
    "ru": "Вы можете внести $20, $50, $100, $200 или кратно $100.",
    "th": "คุณสามารถฝาก $20, $50, $100, $200 หรือทวีคูณของ $100"
},

"deposit_structure": "Your deposit is split into:\n• 30% MSTC\n• 70% USDC (Polygon Network)",

"deposit_blockchain": "The system uses *MEP-20 blockchain contract*.",    
    
"minting_info": {
    "en": "Minting means creating new FNX tokens.",
    "hi": "मिंटिंग का मतलब नए FNX टोकन बनाना है।",
    "mr": "मिंटिंग म्हणजे नवीन FNX टोकन तयार करणे.",
    "bn": "মিন্টিং মানে নতুন FNX টোকেন তৈরি করা।",
    "vi": "Minting là quá trình tạo token FNX mới.",
    "tl": "Ang minting ay paggawa ng bagong FNX tokens.",
    "ru": "Минтинг означает создание новых токенов FNX.",
    "th": "Minting คือการสร้างโทเค็น FNX ใหม่"
},

"minting_time": "After your deposit transaction is completed.",

"minting_location": "The system automatically credits the minted FNX token to your wallet.",    
    
"lp_info": {
    "en": "Liquidity Pool helps smooth trading.",
    "hi": "लिक्विडिटी पूल ट्रेडिंग को आसान बनाता है।",
    "mr": "लिक्विडिटी पूल ट्रेडिंग सोपे करते.",
    "bn": "লিকুইডিটি পুল ট্রেডিং সহজ করে।",
    "vi": "Liquidity Pool giúp giao dịch mượt mà hơn.",
    "tl": "Ang Liquidity Pool ay nagpapadali ng trading.",
    "ru": "Пул ликвидности обеспечивает плавную торговлю.",
    "th": "Liquidity Pool ช่วยให้การซื้อขายราบรื่น"
},

"lp_pair": "FNX + USDC pair is used.",

"lp_benefits": "Stable trading\n• Daily passive income\n• High rewards\n• Community growth\n• Strong ecosystem support",

"lp_rewards": "You can earn daily rewards up to *5% MPY (Monthly Percentage Yield)*.\nThese rewards are generated from the system's trading and ecosystem activity.",
    
"staking_info": {
    "en": "Staking means locking FNX tokens to earn rewards.",
    "hi": "स्टेकिंग का मतलब FNX टोकन लॉक करके रिवॉर्ड कमाना है।",
    "mr": "स्टेकिंग म्हणजे FNX टोकन लॉक करून रिवॉर्ड मिळवणे.",
    "bn": "স্টেকিং মানে FNX টোকেন লক করে রিওয়ার্ড আয় করা।",
    "vi": "Staking là khóa FNX để nhận thưởng.",
    "tl": "Ang staking ay pag-lock ng FNX para kumita.",
    "ru": "Стейкинг — это блокировка FNX для получения наград.",
    "th": "Staking คือการล็อก FNX เพื่อรับรางวัล"
},

"staking_work": "The staking process is very simple:\n• Deposit funds into the platform\n• FNX tokens are minted and credited to your wallet\n• Stake your FNX tokens in the Self-Staking section\n• The system generates daily rewards automatically\n• You can withdraw rewards anytime",

"staking_rewards": "Up to *2% MPY (Monthly Percentage Yield)* daily reward.",    
    
"withdraw_anytime": {
    "en": "Yes, you can withdraw anytime.",
    "hi": "हाँ, आप कभी भी निकाल सकते हैं।",
    "mr": "होय, तुम्ही कधीही पैसे काढू शकता.",
    "bn": "হ্যাঁ, আপনি যেকোনো সময় উত্তোলন করতে পারেন।"
},

"withdraw_currency": "You will receive *USDC* in your wallet instantly.",

"withdraw_burn": "When you withdraw FNX:\n• 50% FNX is burned\n• 50% FNX goes back to supply.\nThis helps control token supply.",    
    
"airdrop_eligibility": "Yes, you must have at least *5 direct paid referrals*.",

"airdrop_reward": "You receive *50 FNX tokens*.",

"airdrop_conditions": "• Wallet must be registered\n• User must be verified\n• Duplicate referrals are not counted",   
    
"affiliate_info": {
    "en": "Affiliate program lets you earn by building a team.",
    "hi": "एफिलिएट प्रोग्राम से आप टीम बनाकर कमाई कर सकते हैं।",
    "mr": "अ‍ॅफिलिएट प्रोग्रामद्वारे टीम बनवून कमाई करता येते.",
    "bn": "অ্যাফিলিয়েট প্রোগ্রাম আপনাকে দল গড়ে আয় করতে দেয়।"
},

"affiliate_team": "The total deposits made by your team.",

"affiliate_importance": "It helps grow the community and increases earnings.",    
    
"rank_structure": {
    "en": "Ranks: Origin, Life Changer, Advisor, Visionary, Creator.",
    "hi": "रैंक: ओरिजिन, लाइफ चेंजर, एडवाइजर, विजनरी, क्रिएटर।",
    "mr": "रँक: ओरिजिन, लाइफ चेंजर, अ‍ॅडव्हायझर, व्हिजनरी, क्रिएटर.",
    "bn": "র‍্যাঙ্ক: অরিজিন, লাইফ চেঞ্জার, অ্যাডভাইজার, ভিশনারি, ক্রিয়েটর।"
},

"club_rewards": "• Rank 1 (Origin) — 10%\n• Rank 2 (Life Changer) — 16% (3% CTO club share)\n• Rank 3 (Advisor) — 20% (2.5% CTO club share)\n• Rank 4 (Visionary) — 23% (2% CTO club share)\n• Rank 5 (Creator) — 25% (1.5% CTO club share)",

"rank_requirements": "• Rank 1 (Origin)\n  • Self activation\n• Rank 2 (Life Changer)\n  • $1000 team business\n  • 10 active origins\n  • Minimum $30 LP\n• Rank 3 (Advisor)\n  • $5000 team business\n  • 2 active life changers\n  • Minimum $100 LP\n• Rank 4 (Visionary)\n  • $25,000 team business\n  • 2 active advisors\n  • Minimum $300 LP\n• Rank 5 (Creator)\n  • $100,000 team business\n  • 2 active visionaries\n  • Minimum $1000 LP",    
    
"triple_info": {
    "en": "Earn from performance, LP and staking.",
    "hi": "परफॉर्मेंस, LP और स्टेकिंग से कमाई करें।",
    "mr": "परफॉर्मन्स, LP आणि स्टेकिंगमधून कमाई करा.",
    "bn": "পারফরম্যান্স, LP এবং স্টেকিং থেকে আয় করুন।"
},


"triple_limit": "After *3x performance income*, you must *retop-up* to continue earning.",    
    
"terms_conditions": "*General T&C*\n• Anyone can join the program\n• Rewards based on company policy\n• Company may update program anytime\n\n*Airdrop T&C*\n• Wallet registration + verification\n• Limited period\n• Duplicate referrals not counted\n\n*Additional T&C*\n• LP 50% counts in team business\n• Performance income limit: 3X\n• Retop-up required after limit\n• Retop-up gives 50% FNX",   
    
"risk_disclaimer": {
    "en": "Crypto investments carry risk.",
    "hi": "क्रिप्टो निवेश में जोखिम होता है।",
    "mr": "क्रिप्टो गुंतवणुकीत जोखीम असते.",
    "bn": "ক্রিপ্টো বিনিয়োগে ঝুঁকি রয়েছে।"
}

   
 }


TRANSLATION_CACHE = {}

# ===================== UI HELPERS =====================


def header_buttons():
    return [
        [{"text": "🌐 Change Language", "callback_data": "change_lang"}],

        [
            {"text": "🚀 Open App", "url": "https://finux-chatbot-production.up.railway.app"},
            {
                "text": "❓ Why Finux ?",
                "web_app": {
                    "url": "https://finux-chatbot-production.up.railway.app/static/why-finux.html"
                }
            }
        ],

        [
            {"text": "📢 Channel", "url": "https://t.me/Finuxofficiallive"},
            {"text": "🌐 Website", "url": "https://finux.online/"}
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
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

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
                        "reply_markup": build_language_menu()
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
                    "reply_markup": build_language_menu()
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