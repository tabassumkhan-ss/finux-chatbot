from docx import Document

def load_docx(path: str) -> list[str]:
    doc = Document(path)
    sections = []

    buffer = []
    for para in doc.paragraphs:
        if para.text.strip():
            buffer.append(para.text.strip())

    sections.append("\n".join(buffer))
    return sections
