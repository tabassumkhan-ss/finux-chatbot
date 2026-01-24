from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks: list[str]):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)
    return db
