from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(texts: list[str]) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))

    return chunks
