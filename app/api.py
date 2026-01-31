from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from embeddings.vector_store import create_vector_store
from embeddings.text_loader import load_finux_text
from embeddings.text_splitter import split_text

from gemini import ask_gemini

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load FINUX Knowledge ONCE
# -------------------------

print("Loading FINUX data...")

raw_text = load_finux_text()
chunks = split_text(raw_text)

db = create_vector_store(chunks)

print("FINUX Vector DB Ready")

# -------------------------

class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "FINUX chatbot running"}


# -------------------------
# RAG
# -------------------------

def rag_answer(question: str):

    docs = db.similarity_search_with_score(question, k=2)

    if not docs:
        return None

    doc, score = docs[0]

    # higher = worse
    if score > 0.75:
        return None

    return doc.page_content


# -------------------------
# Chat Endpoint
# -------------------------

@app.post("/chat")
async def chat(req: ChatRequest):

    question = req.message

    finux_answer = rag_answer(question)

    if finux_answer and finux_answer.strip():
        answer = finux_answer

    else:
        gemini_answer = ask_gemini(question)

        if gemini_answer and gemini_answer.strip():
            answer = gemini_answer
        else:
            answer = "Sorry — I could not generate a reply."

    return {"response": answer}
