import re
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.db import init_db, save_question

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.docx_loader import load_docx
from app.ingestion.chunker import chunk_text
from app.embeddings.vector_store import create_vector_store

app = FastAPI(title="FINUX Chatbot API")


# ---------------- Startup ----------------

@app.on_event("startup")
def on_startup():
    init_db()


# ---------------- UI ----------------

@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/ui.html", "r", encoding="utf-8") as f:
        return f.read()


# ---------------- Knowledge Base ----------------

PDF_PATH = "data/raw/finux.pdf"
DOCX_PATH = "data/raw/finux.docx"

pdf_text = load_pdf(PDF_PATH)
docx_text = load_docx(DOCX_PATH)
chunks = chunk_text(pdf_text + docx_text)
db = create_vector_store(chunks)


# ---------------- Models ----------------

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# ---------------- Chat Endpoint ----------------

@app.post("/chat")
def chat(req: ChatRequest):

    # Save question safely (never break chat)
    try:
        save_question(req.question)
    except Exception as e:
        print("DB save failed:", e)

    q = req.question.lower()

    # Retrieve relevant chunks
    docs = db.similarity_search(req.question, k=3)

    if not docs:
        return {"answer": "Sorry — mujhe FINUX documents me iska jawab nahi mila."}

    # Combine + deduplicate chunks
    raw = "\n\n".join(dict.fromkeys(d.page_content for d in docs))

    # Remove [Page X]
    clean = re.sub(r"\[Page\s*\d+\]", "", raw).strip()

    # Remove duplicate lines
    clean = "\n".join(dict.fromkeys(clean.splitlines()))

    # Limit length (avoid PDF dump)
    clean = clean[:1200]

    # ---------------- Language detection ----------------

    hinglish_markers = [
        " kya ", " mujhe ", " samjha", " bata", " ka ", " ke ",
        " kyun", " kaise", " hai", " hain", " aap", " tum", " mera"
    ]

    q_spaced = f" {q} "

    if re.search(r"[a-zA-Z]", req.question):
        is_hinglish = any(w in q_spaced for w in hinglish_markers)
        is_english = not is_hinglish
    else:
        is_english = False

    # ---------------- Intent based intro ----------------

    if any(x in q for x in ["member", "join", "register", "sign up", "become"]):
        intro = (
            "Yes — of course 🙂 You can become a FINUX member. Here’s the simple joining process:"
            if is_english
            else "Yes — bilkul 🙂 aap FINUX ke member ban sakte hain. Neeche joining ka simple process diya gaya hai:"
        )

    elif "club" in q:
        intro = (
            "Good question 🙂 FINUX Clubs work as a structured reward system. Here’s a simple explanation:"
            if is_english
            else "Achha sawaal 🙂 FINUX me Clubs ek reward-based system hote hain. Simple explanation yeh hai:"
        )

    elif "reward" in q or "referral" in q:
        intro = (
            "FINUX rewards mainly come from referrals, staking, and liquidity participation. Here are the main points:"
            if is_english
            else "FINUX me rewards referral, staking aur liquidity participation se milte hain. Main points yeh rahe:"
        )

    elif "deposit" in q or "start" in q:
        intro = (
            "To get started with FINUX, you first need to make a deposit. Here’s how it works:"
            if is_english
            else "FINUX start karne ke liye pehle deposit karna hota hai. Process kuch is tarah hai:"
        )

    else:
        intro = (
            "Here’s a short explanation based on FINUX documents:"
            if is_english
            else "Yeh FINUX documents ke according short explanation hai:"
        )

    # ---------------- Human bridge ----------------

    bridge = "In simple terms:\n\n" if is_english else "Simple words me samjhein:\n\n"

    # ---------------- Final answer ----------------

    answer = f"""{intro}

{bridge}{clean}
"""

    return {"answer": answer.strip()}
