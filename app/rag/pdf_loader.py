from pypdf import PdfReader
from typing import List


def load_pdf_text(file_path: str) -> List[str]:
    reader = PdfReader(file_path)

    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text)

    return pages
