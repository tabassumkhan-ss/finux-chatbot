from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.docx_loader import load_docx
from app.ingestion.chunker import chunk_text


# ---------- Build FINUX vector DB once ----------

PDF_PATH = "data/raw/finux.pdf"
DOCX_PATH = "data/raw/finux.docx"

pdf_text = load_pdf(PDF_PATH)
docx_text = load_docx(DOCX_PATH)

chunks = chunk_text(pdf_text + docx_text)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_texts(chunks, embeddings)


# ---------- Public functions ----------

def create_vector_store(chunks: list[str]):
    return FAISS.from_texts(chunks, embeddings)


def get_rag_answer(question: str) -> str:
    try:
        docs = db.similarity_search(question, k=3)

        if not docs:
            return ""

        text = "\n\n".join(d.page_content for d in docs)

        # limit size
        text = text[:1500]

        return text.strip()

    except Exception as e:
        print("RAG error:", e)
        return ""
