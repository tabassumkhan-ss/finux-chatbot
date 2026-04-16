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

TRANSLATION_CACHE = {}

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

if DOCUMENT_TEXT:
    doc_embeddings = embedding_model.encode(DOCUMENT_TEXT)

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))
else:
    index = None


def semantic_search(question: str, top_k=2):

    if not index:
        return ""

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

    cache_key = f"{text}:{lang}"

    if cache_key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cache_key]

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
            translated = response.text.strip()
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
            "en": "🎁 Airdrop",
            "hi": "🎁 एयरड्रॉप",
            "mr": "🎁 एअरड्रॉप",
            "bn": "🎁 এয়ারড্রপ",
            "vi": "🎁 Airdrop",
            "tl": "🎁 Airdrop",
            "ru": "🎁 Аирдроп",
            "th": "🎁 แอร์ดรอป"
        },
        "action": "menu:airdrop"
    },
    {
        "label": {
            "en": "🤝 Affiliate Program",
            "hi": "🤝 एफिलिएट प्रोग्राम",
            "mr": "🤝 एफिलिएट प्रोग्राम",
            "bn": "🤝 অ্যাফিলিয়েট প্রোগ্রাম",
            "vi": "🤝 Affiliate",
            "tl": "🤝 Affiliate",
            "ru": "🤝 Партнерская программа",
            "th": "🤝 โปรแกรมแอฟฟิลิเอต"
        },
        "action": "menu:affiliate"
    },
    {
        "label": {
            "en": "🏅 Ranks & Clubs",
            "hi": "🏅 रैंक और क्लब",
            "mr": "🏅 रँक आणि क्लब",
            "bn": "🏅 র‍্যাঙ্ক ও ক্লাব",
            "vi": "🏅 Cấp bậc & Club",
            "tl": "🏅 Ranks & Clubs",
            "ru": "🏅 Ранги и клубы",
            "th": "🏅 ระดับและคลับ"
        },
        "action": "menu:ranks"
    },
    {
        "label": {
            "en": "💎 Triple Income System",
            "hi": "💎 ट्रिपल इनकम सिस्टम",
            "mr": "💎 ट्रिपल इनकम सिस्टम",
            "bn": "💎 ট্রিপল ইনকাম সিস্টেম",
            "vi": "💎 Thu nhập 3 nguồn",
            "tl": "💎 Triple Income",
            "ru": "💎 Тройной доход",
            "th": "💎 ระบบรายได้ 3 ทาง"
        },
        "action": "menu:triple"
    },
    {
        "label": {
            "en": "📜 Terms & Conditions",
            "hi": "📜 नियम और शर्तें",
            "mr": "📜 नियम व अटी",
            "bn": "📜 শর্তাবলী",
            "vi": "📜 Điều khoản",
            "tl": "📜 Terms",
            "ru": "📜 Условия",
            "th": "📜 ข้อกำหนด"
        },
        "action": "q:terms_conditions"
    },
    {
        "label": {
            "en": "⚠️ Risk Disclaimer",
            "hi": "⚠️ जोखिम सूचना",
            "mr": "⚠️ जोखीम सूचना",
            "bn": "⚠️ ঝুঁকি সতর্কতা",
            "vi": "⚠️ Rủi ro",
            "tl": "⚠️ Risk",
            "ru": "⚠️ Риски",
            "th": "⚠️ คำเตือนความเสี่ยง"
        },
        "action": "q:risk_disclaimer"
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
    {
        "label": {
            "en": "💸 Can I withdraw anytime?",
            "hi": "💸 क्या मैं कभी भी निकाल सकता हूँ?",
            "mr": "💸 मी कधीही पैसे काढू शकतो का?",
            "bn": "💸 আমি কি যেকোনো সময় উত্তোলন করতে পারি?",
            "vi": "💸 Tôi có thể rút tiền bất cứ lúc nào không?",
            "tl": "💸 Maaari ba akong mag-withdraw anumang oras?",
            "ru": "💸 Могу ли я выводить средства в любое время?",
            "th": "💸 ฉันสามารถถอนเงินได้ทุกเมื่อหรือไม่?"
        },
        "action": "q:withdraw_anytime"
    },
    {
        "label": {
            "en": "💰 Withdrawal currency",
            "hi": "💰 निकासी मुद्रा",
            "mr": "💰 पैसे काढण्याची चलन",
            "bn": "💰 উত্তোলনের মুদ্রা",
            "vi": "💰 Loại tiền rút",
            "tl": "💰 Currency ng withdrawal",
            "ru": "💰 Валюта вывода",
            "th": "💰 สกุลเงินที่ใช้ถอน"
        },
        "action": "q:withdraw_currency"
    },
    {
        "label": {
            "en": "🔥 Token burning mechanism",
            "hi": "🔥 टोकन बर्निंग मैकेनिज्म",
            "mr": "🔥 टोकन बर्निंग मेकॅनिझम",
            "bn": "🔥 টোকেন বার্নিং প্রক্রিয়া",
            "vi": "🔥 Cơ chế đốt token",
            "tl": "🔥 Mekanismo ng token burning",
            "ru": "🔥 Механизм сжигания токенов",
            "th": "🔥 กลไกการเผาโทเค็น"
        },
        "action": "q:withdraw_burn"
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

AIRDROP_MENU = [
    {
        "label": {
            "en": "🎁 Airdrop eligibility",
            "hi": "🎁 एयरड्रॉप पात्रता",
            "mr": "🎁 एअरड्रॉप पात्रता",
            "bn": "🎁 এয়ারড্রপ যোগ্যতা",
            "vi": "🎁 Điều kiện airdrop",
            "tl": "🎁 Airdrop eligibility",
            "ru": "🎁 Условия аирдропа",
            "th": "🎁 คุณสมบัติรับแอร์ดรอป"
        },
        "action": "q:airdrop_eligibility"
    },
    {
        "label": {
            "en": "🎁 Airdrop reward",
            "hi": "🎁 एयरड्रॉप रिवॉर्ड",
            "mr": "🎁 एअरड्रॉप रिवॉर्ड",
            "bn": "🎁 এয়ারড্রপ পুরস্কার",
            "vi": "🎁 Phần thưởng airdrop",
            "tl": "🎁 Airdrop reward",
            "ru": "🎁 Награда аирдропа",
            "th": "🎁 รางวัลแอร์ดรอป"
        },
        "action": "q:airdrop_reward"
    },
    {
        "label": {
            "en": "📜 Airdrop conditions",
            "hi": "📜 एयरड्रॉप शर्तें",
            "mr": "📜 एअरड्रॉप अटी",
            "bn": "📜 এয়ারড্রপ শর্তাবলী",
            "vi": "📜 Điều kiện airdrop",
            "tl": "📜 Airdrop conditions",
            "ru": "📜 Условия аирдропа",
            "th": "📜 เงื่อนไขแอร์ดรอป"
        },
        "action": "q:airdrop_conditions"
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

AFFILIATE_MENU = [
    {
        "label": {
            "en": "👥 What is the affiliate program?",
            "hi": "👥 एफिलिएट जानकारी",
            "mr": "👥 एफिलिएट माहिती",
            "bn": "👥 অ্যাফিলিয়েট তথ্য",
            "vi": "👥 Thông tin affiliate",
            "tl": "👥 Affiliate info",
            "ru": "👥 Информация об аффилиатах",
            "th": "👥 ข้อมูลแอฟฟิลิเอต"
        },
        "action": "q:affiliate_info"
    },
    {
        "label": {
            "en": "👥 Team business",
            "hi": "👥 टीम बिजनेस",
            "mr": "👥 टीम बिझनेस",
            "bn": "👥 টিম বিজনেস",
            "vi": "👥 Doanh số đội nhóm",
            "tl": "👥 Team business",
            "ru": "👥 Командный оборот",
            "th": "👥 ธุรกิจทีม"
        },
        "action": "q:affiliate_team"
    },
    {
        "label": {
            "en": "📈 Importance",
            "hi": "📈 महत्व",
            "mr": "📈 महत्त्व",
            "bn": "📈 গুরুত্ব",
            "vi": "📈 Tầm quan trọng",
            "tl": "📈 Importance",
            "ru": "📈 Важность",
            "th": "📈 ความสำคัญ"
        },
        "action": "q:affiliate_importance"
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

RANKS_MENU = [
    {
        "label": {
            "en": "🏆 Rank structure",
            "hi": "🏆 रैंक संरचना",
            "mr": "🏆 रँक स्ट्रक्चर",
            "bn": "🏆 র‍্যাঙ্ক স্ট্রাকচার",
            "vi": "🏆 Cấu trúc cấp bậc",
            "tl": "🏆 Rank structure",
            "ru": "🏆 Структура рангов",
            "th": "🏆 โครงสร้างระดับ"
        },
        "action": "q:rank_structure"
    },
    {
        "label": {
            "en": "🎯 Rank requirements",
            "hi": "🎯 रैंक आवश्यकताएं",
            "mr": "🎯 रँक आवश्यकता",
            "bn": "🎯 র‍্যাঙ্ক প্রয়োজনীয়তা",
            "vi": "🎯 Yêu cầu cấp bậc",
            "tl": "🎯 Rank requirements",
            "ru": "🎯 Требования рангов",
            "th": "🎯 ข้อกำหนดระดับ"
        },
        "action": "q:rank_requirements"
    },
    {
        "label": {
            "en": "🎉 Club rewards",
            "hi": "🎉 क्लब रिवॉर्ड",
            "mr": "🎉 क्लब रिवॉर्ड",
            "bn": "🎉 ক্লাব পুরস্কার",
            "vi": "🎉 Phần thưởng club",
            "tl": "🎉 Club rewards",
            "ru": "🎉 Клубные награды",
            "th": "🎉 รางวัลคลับ"
        },
        "action": "q:club_rewards"
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

TRIPLE_MENU = [
    {
        "label": {
            "en": "💰 Triple income",
            "hi": "💰 ट्रिपल इनकम",
            "mr": "💰 ट्रिपल इनकम",
            "bn": "💰 ট্রিপল ইনকাম",
            "vi": "💰 Thu nhập 3 nguồn",
            "tl": "💰 Triple income",
            "ru": "💰 Тройной доход",
            "th": "💰 รายได้ 3 ทาง"
        },
        "action": "q:triple_info"
    },
    {
        "label": {
            "en": "⚠ Income limit",
            "hi": "⚠ आय सीमा",
            "mr": "⚠ उत्पन्न मर्यादा",
            "bn": "⚠ আয়ের সীমা",
            "vi": "⚠ Giới hạn thu nhập",
            "tl": "⚠ Income limit",
            "ru": "⚠ Лимит дохода",
            "th": "⚠ ขีดจำกัดรายได้"
        },
        "action": "q:triple_limit"
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
    "bn": "আপনার প্রাইভেট কি নিরাপদ রাখুন, হারালে ফান্ড ফিরে পাবেন না।",
    "vi": "Hãy giữ khóa riêng của bạn an toàn. Nếu mất, không thể khôi phục tài sản.",
    "tl": "Panatilihing ligtas ang iyong private key. Kapag nawala, hindi na mababawi ang pondo.",
    "ru": "Храните свой приватный ключ в безопасности. При утере средства восстановить невозможно.",
    "th": "เก็บคีย์ส่วนตัวของคุณให้ปลอดภัย หากสูญหายจะไม่สามารถกู้คืนเงินได้"
},

"wallet_private": {
    "en": "Private key gives full control of your wallet. Never share it.",
    "hi": "प्राइवेट की आपके वॉलेट का पूरा नियंत्रण देती है, इसे कभी शेयर न करें।",
    "mr": "प्रायव्हेट की तुमच्या वॉलेटवर पूर्ण नियंत्रण देते, कधीही शेअर करू नका.",
    "bn": "প্রাইভেট কি আপনার ওয়ালেটের সম্পূর্ণ নিয়ন্ত্রণ দেয়, কখনও শেয়ার করবেন না।",
    "vi": "Khóa riêng cho phép toàn quyền kiểm soát ví của bạn. Không bao giờ chia sẻ nó.",
    "tl": "Ang private key ay nagbibigay ng buong kontrol sa iyong wallet. Huwag itong ibahagi kailanman.",
    "ru": "Приватный ключ дает полный контроль над вашим кошельком. Никогда не передавайте его.",
    "th": "คีย์ส่วนตัวให้สิทธิ์ควบคุมกระเป๋าของคุณทั้งหมด ห้ามแชร์กับผู้อื่น"
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

"deposit_structure": {
    "en": "Your deposit is split into:\n• 30% MSTC\n• 70% USDC (Polygon Network)",
    "hi": "आपकी जमा राशि इस प्रकार विभाजित होती है:\n• 30% MSTC\n• 70% USDC (Polygon नेटवर्क)",
    "mr": "तुमची जमा रक्कम अशा प्रकारे विभागली जाते:\n• 30% MSTC\n• 70% USDC (Polygon नेटवर्क)",
    "bn": "আপনার জমা এইভাবে ভাগ করা হয়:\n• 30% MSTC\n• 70% USDC (Polygon নেটওয়ার্ক)",
    "vi": "Khoản nạp của bạn được chia thành:\n• 30% MSTC\n• 70% USDC (Mạng Polygon)",
    "tl": "Ang iyong deposito ay hinahati sa:\n• 30% MSTC\n• 70% USDC (Polygon Network)",
    "ru": "Ваш депозит делится на:\n• 30% MSTC\n• 70% USDC (сеть Polygon)",
    "th": "เงินฝากของคุณจะแบ่งเป็น:\n• 30% MSTC\n• 70% USDC (เครือข่าย Polygon)"
},

"deposit_blockchain": {
    "en": "The system uses *MEP-20 blockchain contract*.",
    "hi": "सिस्टम *MEP-20 ब्लॉकचेन कॉन्ट्रैक्ट* का उपयोग करता है।",
    "mr": "सिस्टम *MEP-20 ब्लॉकचेन कॉन्ट्रॅक्ट* वापरते.",
    "bn": "সিস্টেম *MEP-20 ব্লকচেইন কনট্রাক্ট* ব্যবহার করে।",
    "vi": "Hệ thống sử dụng hợp đồng blockchain *MEP-20*.",
    "tl": "Gumagamit ang system ng *MEP-20 blockchain contract*.",
    "ru": "Система использует контракт блокчейна *MEP-20*.",
    "th": "ระบบใช้สัญญาบล็อกเชน *MEP-20*"
},
    
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

"minting_time": {
    "en": "After your deposit transaction is completed.",
    "hi": "आपकी जमा लेन-देन पूरी होने के बाद।",
    "mr": "तुमचा जमा व्यवहार पूर्ण झाल्यानंतर.",
    "bn": "আপনার জমা লেনদেন সম্পন্ন হওয়ার পর।",
    "vi": "Sau khi giao dịch nạp tiền của bạn hoàn tất.",
    "tl": "Pagkatapos makumpleto ang iyong deposito na transaksyon.",
    "ru": "После завершения вашей депозитной транзакции.",
    "th": "หลังจากธุรกรรมการฝากของคุณเสร็จสมบูรณ์"
},

"minting_location": {
    "en": "The system automatically credits the minted FNX token to your wallet.",
    "hi": "सिस्टम स्वचालित रूप से मिंट किए गए FNX टोकन को आपके वॉलेट में जमा करता है।",
    "mr": "सिस्टम स्वयंचलितपणे मिंट केलेले FNX टोकन तुमच्या वॉलेटमध्ये जमा करते.",
    "bn": "সিস্টেম স্বয়ংক্রিয়ভাবে মিন্ট করা FNX টোকেন আপনার ওয়ালেটে জমা করে।",
    "vi": "Hệ thống tự động ghi nhận token FNX đã mint vào ví của bạn.",
    "tl": "Awtomatikong ilalagay ng system ang minted FNX token sa iyong wallet.",
    "ru": "Система автоматически зачисляет созданные токены FNX в ваш кошелек.",
    "th": "ระบบจะโอนโทเค็น FNX ที่สร้างขึ้นไปยังกระเป๋าของคุณโดยอัตโนมัติ"
},
    
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

"lp_pair": {
    "en": "FNX + USDC pair is used.",
    "hi": "FNX + USDC पेयर का उपयोग किया जाता है।",
    "mr": "FNX + USDC पेअर वापरले जाते.",
    "bn": "FNX + USDC পেয়ার ব্যবহার করা হয়।",
    "vi": "Sử dụng cặp FNX + USDC.",
    "tl": "Ginagamit ang FNX + USDC na pares.",
    "ru": "Используется пара FNX + USDC.",
    "th": "ใช้คู่ FNX + USDC"
},

"lp_benefits": {
    "en": "Stable trading\n• Daily passive income\n• High rewards\n• Community growth\n• Strong ecosystem support",
    "hi": "स्थिर ट्रेडिंग\n• दैनिक निष्क्रिय आय\n• उच्च रिवॉर्ड\n• कम्युनिटी ग्रोथ\n• मजबूत इकोसिस्टम सपोर्ट",
    "mr": "स्थिर ट्रेडिंग\n• दररोज निष्क्रिय उत्पन्न\n• उच्च रिवॉर्ड\n• कम्युनिटी वाढ\n• मजबूत इकोसिस्टम सपोर्ट",
    "bn": "স্থিতিশীল ট্রেডিং\n• দৈনিক প্যাসিভ আয়\n• উচ্চ পুরস্কার\n• কমিউনিটি বৃদ্ধি\n• শক্তিশালী ইকোসিস্টেম সাপোর্ট",
    "vi": "Giao dịch ổn định\n• Thu nhập thụ động hàng ngày\n• Phần thưởng cao\n• Phát triển cộng đồng\n• Hệ sinh thái mạnh mẽ",
    "tl": "Matatag na trading\n• Araw-araw na passive income\n• Mataas na reward\n• Paglago ng komunidad\n• Malakas na ecosystem support",
    "ru": "Стабильная торговля\n• Ежедневный пассивный доход\n• Высокие награды\n• Рост сообщества\n• Сильная экосистема",
    "th": "การซื้อขายที่เสถียร\n• รายได้แบบพาสซีฟรายวัน\n• รางวัลสูง\n• การเติบโตของชุมชน\n• ระบบนิเวศที่แข็งแกร่ง"
},

"lp_rewards": {
    "en": "You can earn daily rewards up to *5% MPY (Monthly Percentage Yield)*.\nThese rewards are generated from the system's trading and ecosystem activity.",
    "hi": "आप दैनिक *5% MPY (Monthly Percentage Yield)* तक रिवॉर्ड कमा सकते हैं।\nये रिवॉर्ड सिस्टम की ट्रेडिंग और इकोसिस्टम गतिविधियों से उत्पन्न होते हैं।",
    "mr": "तुम्ही दररोज *5% MPY (Monthly Percentage Yield)* पर्यंत रिवॉर्ड मिळवू शकता.\nहे रिवॉर्ड सिस्टमच्या ट्रेडिंग आणि इकोसिस्टम क्रियाकलापांमधून मिळतात.",
    "bn": "আপনি দৈনিক *5% MPY (Monthly Percentage Yield)* পর্যন্ত পুরস্কার পেতে পারেন।\nএই পুরস্কার সিস্টেমের ট্রেডিং এবং ইকোসিস্টেম কার্যক্রম থেকে আসে।",
    "vi": "Bạn có thể kiếm phần thưởng hàng ngày lên đến *5% MPY (Monthly Percentage Yield)*.\nPhần thưởng này được tạo ra từ hoạt động giao dịch và hệ sinh thái.",
    "tl": "Maaari kang kumita ng hanggang *5% MPY (Monthly Percentage Yield)* araw-araw.\nAng mga reward na ito ay mula sa trading at ecosystem activity.",
    "ru": "Вы можете получать до *5% MPY (Monthly Percentage Yield)* ежедневно.\nЭти награды генерируются за счет торговой и экосистемной активности.",
    "th": "คุณสามารถรับรางวัลรายวันสูงสุด *5% MPY (Monthly Percentage Yield)*.\nรางวัลเหล่านี้มาจากกิจกรรมการซื้อขายและระบบนิเวศ"
},
    
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

"staking_work": {
    "en": "The staking process is very simple:\n• Deposit funds into the platform\n• FNX tokens are minted and credited to your wallet\n• Stake your FNX tokens in the Self-Staking section\n• The system generates daily rewards automatically\n• You can withdraw rewards anytime",
    "hi": "स्टेकिंग प्रक्रिया बहुत सरल है:\n• प्लेटफॉर्म में फंड जमा करें\n• FNX टोकन मिंट होकर आपके वॉलेट में जमा होते हैं\n• अपने FNX टोकन को सेल्फ-स्टेकिंग सेक्शन में स्टेक करें\n• सिस्टम प्रतिदिन स्वचालित रूप से रिवॉर्ड उत्पन्न करता है\n• आप कभी भी रिवॉर्ड निकाल सकते हैं",
    "mr": "स्टेकिंग प्रक्रिया खूप सोपी आहे:\n• प्लॅटफॉर्ममध्ये निधी जमा करा\n• FNX टोकन मिंट होऊन तुमच्या वॉलेटमध्ये जमा होतात\n• तुमचे FNX टोकन सेल्फ-स्टेकिंग विभागात स्टेक करा\n• सिस्टम दररोज आपोआप रिवॉर्ड तयार करते\n• तुम्ही कधीही रिवॉर्ड काढू शकता",
    "bn": "স্টেকিং প্রক্রিয়া খুব সহজ:\n• প্ল্যাটফর্মে ফান্ড জমা করুন\n• FNX টোকেন মিন্ট হয়ে আপনার ওয়ালেটে জমা হয়\n• আপনার FNX টোকেন সেলফ-স্টেকিং সেকশনে স্টেক করুন\n• সিস্টেম প্রতিদিন স্বয়ংক্রিয়ভাবে পুরস্কার তৈরি করে\n• আপনি যেকোনো সময় পুরস্কার তুলতে পারেন",
    "vi": "Quy trình staking rất đơn giản:\n• Nạp tiền vào nền tảng\n• Token FNX được mint và chuyển vào ví của bạn\n• Stake FNX trong phần Self-Staking\n• Hệ thống tự động tạo phần thưởng hàng ngày\n• Bạn có thể rút phần thưởng bất cứ lúc nào",
    "tl": "Napakasimple ng proseso ng staking:\n• Magdeposito ng pondo sa platform\n• Ang FNX tokens ay mina-mint at inilalagay sa iyong wallet\n• I-stake ang FNX tokens sa Self-Staking section\n• Awtomatikong nagbibigay ng daily rewards ang system\n• Maaari mong i-withdraw ang rewards anumang oras",
    "ru": "Процесс стейкинга очень простой:\n• Внесите средства на платформу\n• Токены FNX создаются и зачисляются в ваш кошелек\n• Разместите FNX в разделе Self-Staking\n• Система автоматически генерирует ежедневные награды\n• Вы можете выводить награды в любое время",
    "th": "กระบวนการ Stake นั้นง่ายมาก:\n• ฝากเงินเข้าสู่แพลตฟอร์ม\n• โทเค็น FNX จะถูกสร้างและโอนเข้ากระเป๋าของคุณ\n• นำ FNX ไป Stake ในส่วน Self-Staking\n• ระบบจะสร้างรางวัลรายวันโดยอัตโนมัติ\n• คุณสามารถถอนรางวัลได้ตลอดเวลา"
},

"staking_rewards": {
    "en": "Up to *2% MPY (Monthly Percentage Yield)* daily reward.",
    "hi": "दैनिक *2% MPY (Monthly Percentage Yield)* तक रिवॉर्ड।",
    "mr": "दररोज *2% MPY (Monthly Percentage Yield)* पर्यंत रिवॉर्ड.",
    "bn": "দৈনিক *2% MPY (Monthly Percentage Yield)* পর্যন্ত পুরস্কার।",
    "vi": "Phần thưởng hàng ngày lên đến *2% MPY (Monthly Percentage Yield)*.",
    "tl": "Hanggang *2% MPY (Monthly Percentage Yield)* na daily reward.",
    "ru": "Ежедневная награда до *2% MPY (Monthly Percentage Yield)*.",
    "th": "รางวัลรายวันสูงสุด *2% MPY (Monthly Percentage Yield)*"
},    
    
"withdraw_anytime": {
    "en": "Yes, FNX rewards can be withdrawn anytime.",
    "hi": "हाँ, FNX रिवॉर्ड कभी भी निकाले जा सकते हैं।",
    "mr": "होय, FNX रिवॉर्ड कधीही काढता येतात.",
    "bn": "হ্যাঁ, FNX পুরস্কার যেকোনো সময় উত্তোলন করা যায়।",
    "vi": "Có, phần thưởng FNX có thể rút bất cứ lúc nào.",
    "tl": "Oo, maaaring i-withdraw ang FNX rewards anumang oras.",
    "ru": "Да, вознаграждения FNX можно выводить в любое время.",
    "th": "ใช่ คุณสามารถถอนรางวัล FNX ได้ทุกเมื่อ"
},

"withdraw_currency": {
    "en": "You will receive *USDC* in your wallet instantly.",
    "hi": "आपको तुरंत अपने वॉलेट में *USDC* प्राप्त होगा।",
    "mr": "तुम्हाला लगेच तुमच्या वॉलेटमध्ये *USDC* मिळेल.",
    "bn": "আপনি আপনার ওয়ালেটে সাথে সাথে *USDC* পাবেন।",
    "vi": "Bạn sẽ nhận được *USDC* ngay lập tức trong ví của mình.",
    "tl": "Makakatanggap ka ng *USDC* agad sa iyong wallet.",
    "ru": "Вы получите *USDC* мгновенно в свой кошелек.",
    "th": "คุณจะได้รับ *USDC* ในกระเป๋าของคุณทันที"
},

"withdraw_burn": {
    "en": "When you withdraw FNX:\n• 50% FNX is burned\n• 50% FNX goes back to supply.\nThis helps control token supply.",
    "hi": "जब आप FNX निकालते हैं:\n• 50% FNX बर्न होता है\n• 50% FNX सप्लाई में वापस जाता है\nइससे टोकन सप्लाई नियंत्रित रहती है।",
    "mr": "जेव्हा तुम्ही FNX काढता:\n• 50% FNX बर्न होतो\n• 50% FNX सप्लायमध्ये परत जातो\nयामुळे टोकन सप्लाय नियंत्रित राहतो.",
    "bn": "আপনি যখন FNX উত্তোলন করেন:\n• 50% FNX বার্ন হয়\n• 50% FNX আবার সাপ্লাইতে ফিরে যায়\nএটি টোকেন সাপ্লাই নিয়ন্ত্রণে সাহায্য করে।",
    "vi": "Khi bạn rút FNX:\n• 50% FNX bị đốt\n• 50% FNX quay lại nguồn cung\nĐiều này giúp kiểm soát nguồn cung token.",
    "tl": "Kapag nag-withdraw ka ng FNX:\n• 50% FNX ay sinusunog\n• 50% FNX ay bumabalik sa supply\nNakakatulong ito sa pagkontrol ng supply.",
    "ru": "При выводе FNX:\n• 50% FNX сжигается\n• 50% FNX возвращается в оборот\nЭто помогает контролировать предложение токена.",
    "th": "เมื่อคุณถอน FNX:\n• 50% FNX จะถูกเผา\n• 50% FNX จะกลับเข้าสู่ระบบ\nสิ่งนี้ช่วยควบคุมปริมาณโทเค็น"
},    
    
"airdrop_eligibility": {
    "en": "Yes, you must have at least *5 direct paid referrals*.",
    "hi": "हाँ, आपके पास कम से कम *5 डायरेक्ट पेड रेफरल* होने चाहिए।",
    "mr": "होय, तुमच्याकडे किमान *5 डायरेक्ट पेड रेफरल* असणे आवश्यक आहे.",
    "bn": "হ্যাঁ, আপনার কমপক্ষে *5টি সরাসরি পেইড রেফারেল* থাকতে হবে।",
    "vi": "Có, bạn cần ít nhất *5 người giới thiệu trực tiếp đã thanh toán*.",
    "tl": "Oo, kailangan mong magkaroon ng *5 direktang paid referrals*.",
    "ru": "Да, у вас должно быть как минимум *5 прямых платных рефералов*.",
    "th": "ใช่ คุณต้องมี *ผู้แนะนำโดยตรงที่ชำระเงินอย่างน้อย 5 คน*"
},

"airdrop_reward": {
    "en": "You receive *50 FNX tokens*.",
    "hi": "आपको *50 FNX टोकन* मिलते हैं।",
    "mr": "तुम्हाला *50 FNX टोकन* मिळतात.",
    "bn": "আপনি *50 FNX টোকেন* পাবেন।",
    "vi": "Bạn nhận được *50 FNX token*.",
    "tl": "Makakatanggap ka ng *50 FNX tokens*.",
    "ru": "Вы получите *50 FNX токенов*.",
    "th": "คุณจะได้รับ *50 FNX โทเค็น*"
},

"airdrop_conditions": {
    "en": "• Wallet must be registered\n• User must be verified\n• Duplicate referrals are not counted",
    "hi": "• वॉलेट रजिस्टर होना चाहिए\n• यूजर वेरिफाइड होना चाहिए\n• डुप्लिकेट रेफरल मान्य नहीं होंगे",
    "mr": "• वॉलेट नोंदणीकृत असणे आवश्यक आहे\n• वापरकर्ता सत्यापित असावा\n• डुप्लिकेट रेफरल मोजले जाणार नाहीत",
    "bn": "• ওয়ালেট রেজিস্টার থাকতে হবে\n• ইউজার ভেরিফাইড হতে হবে\n• ডুপ্লিকেট রেফারেল গণনা করা হবে না",
    "vi": "• Ví phải được đăng ký\n• Người dùng phải được xác minh\n• Không tính referral trùng lặp",
    "tl": "• Dapat nakarehistro ang wallet\n• Dapat verified ang user\n• Hindi binibilang ang duplicate referrals",
    "ru": "• Кошелек должен быть зарегистрирован\n• Пользователь должен быть подтвержден\n• Дубликаты рефералов не учитываются",
    "th": "• ต้องลงทะเบียนกระเป๋าเงิน\n• ผู้ใช้ต้องได้รับการยืนยัน\n• จะไม่นับการแนะนำซ้ำ"
},   
    
"affiliate_info": {
    "en": "It is a referral program where you earn rewards by building a team.",
    "hi": "यह एक रेफरल प्रोग्राम है जहाँ आप टीम बनाकर रिवॉर्ड कमाते हैं।",
    "mr": "हा एक रेफरल प्रोग्राम आहे ज्यामध्ये तुम्ही टीम तयार करून रिवॉर्ड मिळवता.",
    "bn": "এটি একটি রেফারেল প্রোগ্রাম যেখানে আপনি টিম তৈরি করে আয় করেন।",
    "vi": "Đây là chương trình giới thiệu nơi bạn kiếm thưởng bằng cách xây dựng đội nhóm.",
    "tl": "Ito ay referral program kung saan kumikita ka sa pagbuo ng team.",
    "ru": "Это реферальная программа, где вы зарабатываете, создавая команду.",
    "th": "นี่คือโปรแกรมแนะนำที่คุณสามารถสร้างรายได้จากการสร้างทีม"
},

"affiliate_team": {
    "en": "The total deposits made by your team.",
    "hi": "आपकी टीम द्वारा किए गए कुल जमा।",
    "mr": "तुमच्या टीमने केलेली एकूण जमा.",
    "bn": "আপনার টিমের মোট জমা।",
    "vi": "Tổng số tiền gửi của đội nhóm bạn.",
    "tl": "Kabuuang deposito ng iyong team.",
    "ru": "Общий депозит вашей команды.",
    "th": "ยอดฝากรวมของทีมคุณ"
},

"affiliate_importance": {
    "en": "It helps grow the community and increases earnings.",
    "hi": "यह कम्युनिटी बढ़ाने और आय बढ़ाने में मदद करता है।",
    "mr": "हे कम्युनिटी वाढवते आणि उत्पन्न वाढवते.",
    "bn": "এটি কমিউনিটি বৃদ্ধি এবং আয় বাড়াতে সাহায্য করে।",
    "vi": "Giúp phát triển cộng đồng và tăng thu nhập.",
    "tl": "Nakakatulong ito sa paglago ng komunidad at kita.",
    "ru": "Помогает развивать сообщество и увеличивать доход.",
    "th": "ช่วยเพิ่มการเติบโตของชุมชนและรายได้"
},    
    
"rank_structure": {
    "en": "• Rank 1 — Origin 10%\n• Rank 2 — Life Changer 16%\n• Rank 3 — Advisor 20%\n• Rank 4 — Visionary 23%\n• Rank 5 — Creator 25%",
    "hi": "• रैंक 1 — ओरिजिन 10%\n• रैंक 2 — लाइफ चेंजर 16%\n• रैंक 3 — एडवाइजर 20%\n• रैंक 4 — विजनरी 23%\n• रैंक 5 — क्रिएटर 25%",
    "mr": "• रँक 1 — ओरिजिन 10%\n• रँक 2 — लाइफ चेंजर 16%\n• रँक 3 — अ‍ॅडव्हायझर 20%\n• रँक 4 — व्हिजनरी 23%\n• रँक 5 — क्रिएटर 25%",
    "bn": "• র‍্যাঙ্ক 1 — অরিজিন 10%\n• র‍্যাঙ্ক 2 — লাইফ চেঞ্জার 16%\n• র‍্যাঙ্ক 3 — অ্যাডভাইজার 20%\n• র‍্যাঙ্ক 4 — ভিশনারি 23%\n• র‍্যাঙ্ক 5 — ক্রিয়েটর 25%",
    "vi": "• Hạng 1 — Origin 10%\n• Hạng 2 — Life Changer 16%\n• Hạng 3 — Advisor 20%\n• Hạng 4 — Visionary 23%\n• Hạng 5 — Creator 25%",
    "tl": "• Rank 1 — Origin 10%\n• Rank 2 — Life Changer 16%\n• Rank 3 — Advisor 20%\n• Rank 4 — Visionary 23%\n• Rank 5 — Creator 25%",
    "ru": "• Ранг 1 — Origin 10%\n• Ранг 2 — Life Changer 16%\n• Ранг 3 — Advisor 20%\n• Ранг 4 — Visionary 23%\n• Ранг 5 — Creator 25%",
    "th": "• ระดับ 1 — Origin 10%\n• ระดับ 2 — Life Changer 16%\n• ระดับ 3 — Advisor 20%\n• ระดับ 4 — Visionary 23%\n• ระดับ 5 — Creator 25%"
},

"club_rewards": {
    "en": "• Rank 1 (Origin) — 10%\n• Rank 2 (Life Changer) — 16% (3% CTO club share)\n• Rank 3 (Advisor) — 20% (2.5% CTO club share)\n• Rank 4 (Visionary) — 23% (2% CTO club share)\n• Rank 5 (Creator) — 25% (1.5% CTO club share)",
    "hi": "• रैंक 1 (ओरिजिन) — 10%\n• रैंक 2 (लाइफ चेंजर) — 16% (3% CTO क्लब शेयर)\n• रैंक 3 (एडवाइजर) — 20% (2.5% CTO क्लब शेयर)\n• रैंक 4 (विजनरी) — 23% (2% CTO क्लब शेयर)\n• रैंक 5 (क्रिएटर) — 25% (1.5% CTO क्लब शेयर)",
    "mr": "• रँक 1 (ओरिजिन) — 10%\n• रँक 2 (लाइफ चेंजर) — 16% (3% CTO क्लब शेअर)\n• रँक 3 (अ‍ॅडव्हायझर) — 20% (2.5% CTO क्लब शेअर)\n• रँक 4 (व्हिजनरी) — 23% (2% CTO क्लब शेअर)\n• रँक 5 (क्रिएटर) — 25% (1.5% CTO क्लब शेअर)",
    "bn": "• র‍্যাঙ্ক 1 (অরিজিন) — 10%\n• র‍্যাঙ্ক 2 (লাইফ চেঞ্জার) — 16% (3% CTO ক্লাব শেয়ার)\n• র‍্যাঙ্ক 3 (অ্যাডভাইজার) — 20% (2.5% CTO ক্লাব শেয়ার)\n• র‍্যাঙ্ক 4 (ভিশনারি) — 23% (2% CTO ক্লাব শেয়ার)\n• র‍্যাঙ্ক 5 (ক্রিয়েটর) — 25% (1.5% CTO ক্লাব শেয়ার)",
    "vi": "• Hạng 1 (Origin) — 10%\n• Hạng 2 (Life Changer) — 16% (3% chia sẻ CTO club)\n• Hạng 3 (Advisor) — 20% (2.5% chia sẻ CTO club)\n• Hạng 4 (Visionary) — 23% (2% chia sẻ CTO club)\n• Hạng 5 (Creator) — 25% (1.5% chia sẻ CTO club)",
    "tl": "• Rank 1 (Origin) — 10%\n• Rank 2 (Life Changer) — 16% (3% CTO club share)\n• Rank 3 (Advisor) — 20% (2.5% CTO club share)\n• Rank 4 (Visionary) — 23% (2% CTO club share)\n• Rank 5 (Creator) — 25% (1.5% CTO club share)",
    "ru": "• Ранг 1 (Origin) — 10%\n• Ранг 2 (Life Changer) — 16% (3% доля CTO клуба)\n• Ранг 3 (Advisor) — 20% (2.5% доля CTO клуба)\n• Ранг 4 (Visionary) — 23% (2% доля CTO клуба)\n• Ранг 5 (Creator) — 25% (1.5% доля CTO клуба)",
    "th": "• ระดับ 1 (Origin) — 10%\n• ระดับ 2 (Life Changer) — 16% (ส่วนแบ่ง CTO club 3%)\n• ระดับ 3 (Advisor) — 20% (ส่วนแบ่ง CTO club 2.5%)\n• ระดับ 4 (Visionary) — 23% (ส่วนแบ่ง CTO club 2%)\n• ระดับ 5 (Creator) — 25% (ส่วนแบ่ง CTO club 1.5%)"
},

"rank_requirements": {
    "en": "• Rank 1 (Origin)\n  • Self activation\n• Rank 2 (Life Changer)\n  • $1000 team business\n  • 10 active origins\n  • Minimum $30 LP\n• Rank 3 (Advisor)\n  • $5000 team business\n  • 2 active life changers\n  • Minimum $100 LP\n• Rank 4 (Visionary)\n  • $25,000 team business\n  • 2 active advisors\n  • Minimum $300 LP\n• Rank 5 (Creator)\n  • $100,000 team business\n  • 2 active visionaries\n  • Minimum $1000 LP",
    
    "hi": "• रैंक 1 (ओरिजिन)\n  • सेल्फ एक्टिवेशन\n• रैंक 2 (लाइफ चेंजर)\n  • $1000 टीम बिजनेस\n  • 10 एक्टिव ओरिजिन\n  • न्यूनतम $30 LP\n• रैंक 3 (एडवाइजर)\n  • $5000 टीम बिजनेस\n  • 2 एक्टिव लाइफ चेंजर\n  • न्यूनतम $100 LP\n• रैंक 4 (विजनरी)\n  • $25,000 टीम बिजनेस\n  • 2 एक्टिव एडवाइजर\n  • न्यूनतम $300 LP\n• रैंक 5 (क्रिएटर)\n  • $100,000 टीम बिजनेस\n  • 2 एक्टिव विजनरी\n  • न्यूनतम $1000 LP",
    
    "mr": "• रँक 1 (ओरिजिन)\n  • सेल्फ अ‍ॅक्टिवेशन\n• रँक 2 (लाइफ चेंजर)\n  • $1000 टीम बिझनेस\n  • 10 सक्रिय ओरिजिन\n  • किमान $30 LP\n• रँक 3 (अ‍ॅडव्हायझर)\n  • $5000 टीम बिझनेस\n  • 2 सक्रिय लाइफ चेंजर\n  • किमान $100 LP\n• रँक 4 (व्हिजनरी)\n  • $25,000 टीम बिझनेस\n  • 2 सक्रिय अ‍ॅडव्हायझर\n  • किमान $300 LP\n• रँक 5 (क्रिएटर)\n  • $100,000 टीम बिझनेस\n  • 2 सक्रिय व्हिजनरी\n  • किमान $1000 LP",
    
    "bn": "• র‍্যাঙ্ক 1 (অরিজিন)\n  • সেলফ অ্যাক্টিভেশন\n• র‍্যাঙ্ক 2 (লাইফ চেঞ্জার)\n  • $1000 টিম বিজনেস\n  • 10 সক্রিয় অরিজিন\n  • ন্যূনতম $30 LP\n• র‍্যাঙ্ক 3 (অ্যাডভাইজার)\n  • $5000 টিম বিজনেস\n  • 2 সক্রিয় লাইফ চেঞ্জার\n  • ন্যূনতম $100 LP\n• র‍্যাঙ্ক 4 (ভিশনারি)\n  • $25,000 টিম বিজনেস\n  • 2 সক্রিয় অ্যাডভাইজার\n  • ন্যূনতম $300 LP\n• র‍্যাঙ্ক 5 (ক্রিয়েটর)\n  • $100,000 টিম বিজনেস\n  • 2 সক্রিয় ভিশনারি\n  • ন্যূনতম $1000 LP",
    
    "vi": "• Hạng 1 (Origin)\n  • Kích hoạt cá nhân\n• Hạng 2 (Life Changer)\n  • $1000 doanh số đội nhóm\n  • 10 Origin hoạt động\n  • Tối thiểu $30 LP\n• Hạng 3 (Advisor)\n  • $5000 doanh số đội nhóm\n  • 2 Life Changer hoạt động\n  • Tối thiểu $100 LP\n• Hạng 4 (Visionary)\n  • $25,000 doanh số đội nhóm\n  • 2 Advisor hoạt động\n  • Tối thiểu $300 LP\n• Hạng 5 (Creator)\n  • $100,000 doanh số đội nhóm\n  • 2 Visionary hoạt động\n  • Tối thiểu $1000 LP",
    
    "tl": "• Rank 1 (Origin)\n  • Self activation\n• Rank 2 (Life Changer)\n  • $1000 team business\n  • 10 active origins\n  • Minimum $30 LP\n• Rank 3 (Advisor)\n  • $5000 team business\n  • 2 active life changers\n  • Minimum $100 LP\n• Rank 4 (Visionary)\n  • $25,000 team business\n  • 2 active advisors\n  • Minimum $300 LP\n• Rank 5 (Creator)\n  • $100,000 team business\n  • 2 active visionaries\n  • Minimum $1000 LP",
    
    "ru": "• Ранг 1 (Origin)\n  • Самоактивация\n• Ранг 2 (Life Changer)\n  • $1000 командный оборот\n  • 10 активных Origin\n  • Минимум $30 LP\n• Ранг 3 (Advisor)\n  • $5000 командный оборот\n  • 2 активных Life Changer\n  • Минимум $100 LP\n• Ранг 4 (Visionary)\n  • $25,000 командный оборот\n  • 2 активных Advisor\n  • Минимум $300 LP\n• Ранг 5 (Creator)\n  • $100,000 командный оборот\n  • 2 активных Visionary\n  • Минимум $1000 LP",
    
    "th": "• ระดับ 1 (Origin)\n  • เปิดใช้งานตัวเอง\n• ระดับ 2 (Life Changer)\n  • ธุรกิจทีม $1000\n  • Origin ที่ใช้งาน 10 คน\n  • ขั้นต่ำ $30 LP\n• ระดับ 3 (Advisor)\n  • ธุรกิจทีม $5000\n  • Life Changer ที่ใช้งาน 2 คน\n  • ขั้นต่ำ $100 LP\n• ระดับ 4 (Visionary)\n  • ธุรกิจทีม $25,000\n  • Advisor ที่ใช้งาน 2 คน\n  • ขั้นต่ำ $300 LP\n• ระดับ 5 (Creator)\n  • ธุรกิจทีม $100,000\n  • Visionary ที่ใช้งาน 2 คน\n  • ขั้นต่ำ $1000 LP"
},    
    
"triple_info": {
    "en": "Users can earn from three sources:\n• Performance income — up to 3x\n• Liquidity pool reward — up to 3x\n• FNX staking — up to 2x",
    "hi": "यूजर तीन स्रोतों से कमा सकते हैं:\n• परफॉर्मेंस इनकम — 3x तक\n• LP रिवॉर्ड — 3x तक\n• स्टेकिंग — 2x तक",
    "mr": "वापरकर्ते तीन स्रोतांमधून कमाई करू शकतात:\n• परफॉर्मन्स इनकम — 3x पर्यंत\n• LP रिवॉर्ड — 3x पर्यंत\n• स्टेकिंग — 2x पर्यंत",
    "bn": "ব্যবহারকারীরা তিনটি উৎস থেকে আয় করতে পারে:\n• পারফরম্যান্স ইনকাম — 3x পর্যন্ত\n• LP রিওয়ার্ড — 3x পর্যন্ত\n• স্টেকিং — 2x পর্যন্ত",
    "vi": "Người dùng có thể kiếm từ 3 nguồn:\n• Thu nhập hiệu suất — tối đa 3x\n• Phần thưởng LP — tối đa 3x\n• Staking — tối đa 2x",
    "tl": "Maaaring kumita mula sa 3 sources:\n• Performance income — hanggang 3x\n• LP reward — hanggang 3x\n• Staking — hanggang 2x",
    "ru": "Пользователи могут зарабатывать из 3 источников:\n• Доход от активности — до 3x\n• Награды LP — до 3x\n• Стейкинг — до 2x",
    "th": "ผู้ใช้สามารถสร้างรายได้จาก 3 แหล่ง:\n• รายได้จากผลงาน — สูงสุด 3 เท่า\n• รางวัล LP — สูงสุด 3 เท่า\n• Staking — สูงสุด 2 เท่า"
},


"triple_limit": {
    "en": "After *3x performance income*, you must *retop-up* to continue earning.",
    "hi": "*3x परफॉर्मेंस इनकम* के बाद, आपको कमाई जारी रखने के लिए *रीटॉप-अप* करना होगा।",
    "mr": "*3x परफॉर्मन्स इनकम* नंतर, कमाई सुरू ठेवण्यासाठी तुम्हाला *रीटॉप-अप* करावे लागेल.",
    "bn": "*3x পারফরম্যান্স ইনকাম* এর পর, আয় চালিয়ে যেতে আপনাকে *রিটপ-আপ* করতে হবে।",
    "vi": "Sau khi đạt *3x thu nhập hiệu suất*, bạn phải *nạp lại (retop-up)* để tiếp tục kiếm tiền.",
    "tl": "Pagkatapos ng *3x performance income*, kailangan mong *mag-retop-up* upang magpatuloy sa pag-earn.",
    "ru": "После достижения *3x дохода от производительности*, необходимо *сделать повторное пополнение (retop-up)*, чтобы продолжить заработок.",
    "th": "หลังจากได้ *รายได้ 3 เท่า (3x)* คุณต้อง *เติมเงินใหม่ (retop-up)* เพื่อรับรายได้ต่อ"
},    
    
"terms_conditions": {
    "en": "General Terms:\n• Anyone can join\n• Rewards depend on policy\n• Company may update anytime",
    "hi": "सामान्य नियम:\n• कोई भी जुड़ सकता है\n• रिवॉर्ड नीति पर निर्भर\n• कंपनी बदलाव कर सकती है",
    "mr": "सामान्य अटी:\n• कोणीही सहभागी होऊ शकतो\n• रिवॉर्ड धोरणावर अवलंबून\n• कंपनी बदल करू शकते",
    "bn": "সাধারণ শর্ত:\n• যে কেউ যোগ দিতে পারে\n• পুরস্কার নীতির উপর নির্ভরশীল\n• কোম্পানি পরিবর্তন করতে পারে",
    "vi": "Điều khoản chung:\n• Ai cũng có thể tham gia\n• Phần thưởng theo chính sách\n• Công ty có thể thay đổi",
    "tl": "General terms:\n• Maaaring sumali ang lahat\n• Depende sa policy ang rewards\n• Maaaring magbago ang kumpanya",
    "ru": "Общие условия:\n• Любой может присоединиться\n• Награды зависят от политики\n• Компания может вносить изменения",
    "th": "ข้อกำหนดทั่วไป:\n• ทุกคนสามารถเข้าร่วมได้\n• รางวัลขึ้นอยู่กับนโยบาย\n• บริษัทสามารถเปลี่ยนแปลงได้"
},

"risk_disclaimer": {
    "en": "Crypto investments carry risk. Earnings are not guaranteed.",
    "hi": "क्रिप्टो निवेश में जोखिम होता है। कमाई की गारंटी नहीं है।",
    "mr": "क्रिप्टो गुंतवणुकीत जोखीम असते. कमाईची हमी नाही.",
    "bn": "ক্রিপ্টো বিনিয়োগে ঝুঁকি রয়েছে। আয়ের নিশ্চয়তা নেই।",
    "vi": "Đầu tư crypto có rủi ro. Thu nhập không được đảm bảo.",
    "tl": "May risk ang crypto investment. Walang garantiya sa kita.",
    "ru": "Криптоинвестиции связаны с риском. Доход не гарантирован.",
    "th": "การลงทุนคริปโตมีความเสี่ยง และไม่รับประกันรายได้"
}

   
 }

KEY_TO_MENU = {
    # Wallet
    "wallet_info": "wallet",
    "wallet_create": "wallet",
    "wallet_security": "wallet",
    "wallet_private": "wallet",

    # Deposit
    "deposit_min": "deposit",
    "deposit_plans": "deposit",
    "deposit_structure": "deposit",
    "deposit_blockchain": "deposit",

    # Minting
    "minting_info": "minting",
    "minting_time": "minting",
    "minting_location": "minting",

    # Liquidity Pool
    "lp_info": "lp",
    "lp_pair": "lp",
    "lp_benefits": "lp",
    "lp_rewards": "lp",

    # Staking
    "staking_info": "staking",
    "staking_work": "staking",
    "staking_rewards": "staking",

    # Withdraw
    "withdraw_anytime": "withdraw",
    "withdraw_currency": "withdraw",
    "withdraw_burn": "withdraw",

    # Airdrop
    "airdrop_eligibility": "airdrop",
    "airdrop_reward": "airdrop",
    "airdrop_conditions": "airdrop",

    # Affiliate
    "affiliate_info": "affiliate",
    "affiliate_team": "affiliate",
    "affiliate_importance": "affiliate",

    # Ranks
    "rank_structure": "ranks",
    "rank_requirements": "ranks",
    "club_rewards": "ranks",

    # Triple
    "triple_info": "triple",
    "triple_limit": "triple",

    # Others (direct answers)
    "terms_conditions": "others",
    "risk_disclaimer": "others",
}


TRANSLATION_CACHE = {}

# ===================== UI HELPERS =====================


def header_buttons():
    return [
        [{"text": "🌐 Change Language", "callback_data": "change_lang"}],

        [
            {"text": "🚀 Open App", "url": "https://finux-chatbot-production.up.railway.app"},
            {
                "text": "❓ Why Finux?",
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

USER_LANG_CACHE = {}

def get_user_language(user_id):
    if user_id in USER_LANG_CACHE:
        return USER_LANG_CACHE[user_id]

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT language FROM users WHERE username=%s",
        (user_id,)
    )

    row = cur.fetchone()

    cur.close()
    conn.close()

    lang = row[0] if row else "en"
    USER_LANG_CACHE[user_id] = lang

    return lang

def get_full_answer(key, user_id):
    data = HARDCODED_ANSWERS.get(key)

    if not data:
        return None

    lang = get_user_language(user_id)

    # ✅ IMPORTANT FIX
    if isinstance(data, dict):
        return data.get(lang, data.get("en"))

    return data


MENU_CACHE = {}

def build_menu(menu_key, user_id=None):

    # 🌐 Get language
    lang = "en"
    if user_id:
        lang = get_user_language(user_id)

    # ⚡ CACHE KEY
    cache_key = f"{menu_key}:{lang}"

    # ⚡ RETURN FROM CACHE
    if cache_key in MENU_CACHE:
        return MENU_CACHE[cache_key]

    # 🧱 Build menu
    keyboard = header_buttons()
    menu_items = get_full_menu(menu_key)

    row = []

    for item in menu_items:

        label_dict = item.get("label", {})
        action = item.get("action", "")

        # 🌍 Safe language fallback
        text = label_dict.get(lang) or label_dict.get("en") or "Option"

        row.append({
            "text": text,
            "callback_data": action
        })

        if len(row) == 2:
            keyboard.append(row)
            row = []

    if row:
        keyboard.append(row)

    result = {"inline_keyboard": keyboard}

    # ⚡ SAVE TO CACHE
    MENU_CACHE[cache_key] = result

    return result

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
    if lang != "en":
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
    try:
        data = await request.json()
        logging.info("Telegram update received")

        async with httpx.AsyncClient(timeout=30) as client:

            # ================= CALLBACK HANDLER =================
            if "callback_query" in data:
                cq = data["callback_query"]
                chat_id = cq["message"]["chat"]["id"]
                payload = cq.get("data", "")

                await client.post(
                    f"{TELEGRAM_API}/answerCallbackQuery",
                    json={"callback_query_id": cq["id"]},
                )

                # 🌐 LANGUAGE SELECT
                if payload.startswith("lang:"):
                    lang = payload.split(":")[1]

                    conn = get_conn()
                    cur = conn.cursor()

                    cur.execute(
                        "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING",
                        (str(chat_id), "telegram_user", "user")
                    )

                    cur.execute(
                        "UPDATE users SET language=%s WHERE username=%s",
                        (lang, str(chat_id))
                    )

                    conn.commit()
                    cur.close()
                    conn.close()

                    MENU_CACHE.clear()
                    USER_LANG_CACHE.pop(str(chat_id), None)

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

                # 🌐 CHANGE LANGUAGE
                if payload == "change_lang":
                    await client.post(
                        f"{TELEGRAM_API}/editMessageText",
                        json={
                            "chat_id": chat_id,
                            "message_id": cq["message"]["message_id"],
                            "text": "🌐 *Please choose your language:*",
                            "parse_mode": "Markdown",
                            "reply_markup": build_language_menu(),
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
                    print("KEY:", key)

                    answer = get_full_answer(key, str(chat_id))

                    if not answer or not str(answer).strip():
                       topic = key.replace("_", " ")
                       answer = semantic_search(topic)

                    if not answer or not str(answer).strip():
                       topic = key.replace("_", " ")
                       answer = generate_answer(topic)

                    if not answer or not str(answer).strip():
                     answer = "No information available."
                    print("FINAL ANSWER:", repr(answer))

                    lang = get_user_language(str(chat_id))
                    if key not in HARDCODED_ANSWERS and lang != "en":
                        answer = translate(answer, lang)

                    message_id = cq["message"]["message_id"]
                    menu_to_show = KEY_TO_MENU.get(key, "main")

                    await client.post(
                        f"{TELEGRAM_API}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": answer,
                            
                        },
                    )
                    return {"ok": True}

            # ================= NORMAL MESSAGE =================
            if "message" in data:
                msg = data["message"]
                chat_id = msg["chat"]["id"]
                text = msg.get("text", "").strip()

                # ✅ FIXED /start BLOCK
                if text == "/start":
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
                            "reply_markup": build_language_menu(),
                        },
                    )

                    return {"ok": True}

                #  AI RESPONSE
                if text:
                    answer = generate_answer(text)

                    lang = get_user_language(str(chat_id))
                    if lang != "en":
                        answer = translate(answer, lang)

                    await client.post(
                        f"{TELEGRAM_API}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": answer,
                            "parse_mode": "Markdown",
                        },
                    )

                    return {"ok": True}

        return {"ok": True}

    except Exception as e:
        logging.error(f"Webhook error: {e}")
        return {"ok": True}