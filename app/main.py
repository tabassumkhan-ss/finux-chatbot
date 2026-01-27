from dotenv import load_dotenv
import os
load_dotenv()

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.docx_loader import load_docx
from app.ingestion.chunker import chunk_text
from app.embeddings.vector_store import create_vector_store

PDF_PATH = "data/raw/finux.pdf"
DOCX_PATH = "data/raw/finux.docx"

def build_knowledge_base():
    pdf_text = load_pdf(PDF_PATH)
    docx_text = load_docx(DOCX_PATH)

    all_text = pdf_text + docx_text
    chunks = chunk_text(all_text)

    vector_db = create_vector_store(chunks)
    return vector_db

if __name__ == "__main__":
    db = build_knowledge_base()
    print(f"Knowledge base built with {db.index.ntotal} vectors")
