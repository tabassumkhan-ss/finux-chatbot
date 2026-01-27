import re
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.docx_loader import load_docx
from app.ingestion.chunker import chunk_text
from app.embeddings.vector_store import create_vector_store

from app.core.prompt import SYSTEM_PROMPT

app = FastAPI(title="FINUX Chatbot API")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/ui.html", "r", encoding="utf-8") as f:
        return f.read()


PDF_PATH = "data/raw/finux.pdf"
DOCX_PATH = "data/raw/finux.docx"

# Build knowledge base ONCE at startup
pdf_text = load_pdf(PDF_PATH)
docx_text = load_docx(DOCX_PATH)
chunks = chunk_text(pdf_text + docx_text)
db = create_vector_store(chunks)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat")
def chat(req: ChatRequest):
    import re

    # Retrieve relevant FINUX chunks (keep small for cleanliness)
    docs = db.similarity_search(req.question, k=3)

    if not docs:
        return {"answer": "Sorry — mujhe FINUX documents me iska jawab nahi mila."}

    # Combine + deduplicate chunks
    raw = "\n\n".join(dict.fromkeys(d.page_content for d in docs))

    # Remove [Page X]
    clean = re.sub(r"\[Page\s*\d+\]", "", raw).strip()

    # Remove duplicate lines
    clean = "\n".join(dict.fromkeys(clean.splitlines()))

    # Limit length to avoid PDF dump
    clean = clean[:1200]

    q = req.question.lower()

    # Simple language detection
    english_words = ["what", "how", "can", "is", "are", "do", "does", "member", "join", "deposit"]
    is_english = sum(w in q for w in english_words) >= 2

    # Intent-based conversational intro
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
    elif "reward" in q:
        intro = (
            "FINUX rewards are based on staking and liquidity participation. Here are the details:"
            if is_english
            else "FINUX me rewards staking aur liquidity participation se milte hain. Details yeh rahi:"
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

    answer = f"""{intro}

{clean}
"""

    return {"answer": answer.strip()}
