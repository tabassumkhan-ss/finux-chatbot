import pdfplumber

def load_pdf(path: str) -> list[str]:
    pages = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append(f"[Page {i+1}]\n{text.strip()}")

    return pages
