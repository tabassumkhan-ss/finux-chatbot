from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from app.embeddings.vector_store import create_vector_store
from app.llm.gemini import ask_gemini

app = FastAPI()

# -----------------------------
# CORS (optional but useful)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load UI
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
UI_PATH = os.path.join(BASE_DIR, "ui.html")

# -----------------------------
# Load FINUX documents
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "finux.txt")

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        texts = [f.read()]
else:
    texts = ["FINUX is a decentralized Crypto Ecosystem."]

# -----------------------------
# Build Vector DB ONCE
# -----------------------------
db = create_vector_store(texts)

# -----------------------------
# Request schema
# -----------------------------
class ChatRequest(BaseModel):
    message: str

# -----------------------------
# Home = UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    with open(UI_PATH, "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
# RAG Answer
# -----------------------------
def rag_answer(question: str) -> str:
    docs = db.similarity_search_with_score(question, k=2)

    if not docs:
        return ""

    context = "\n".join([d[0].page_content for d in docs])

    prompt = f"""
Use the context below to answer.

Context:
{context}

Question:
{question}
"""

    return ask_gemini(prompt)

# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.message

    # 1️⃣ Try FINUX RAG
    finux_answer = rag_answer(question)

    if finux_answer and finux_answer.strip():
        answer = finux_answer
    else:
        # 2️⃣ Fallback Gemini
        gemini_answer = ask_gemini(question)

        if gemini_answer and gemini_answer.strip():
            answer = gemini_answer
        else:
            answer = "Sorry — I could not generate a reply."

    return {"response": answer}

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "FINUX chatbot running"}
