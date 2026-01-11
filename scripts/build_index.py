import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.pipeline import build_index_from_texts
from app.rag.vector_store import FaissVectorStore
from app.rag.pdf_loader import load_pdf_text


# -------------------------
# Load PDFs
# -------------------------
PDF_DIR = ROOT_DIR / "data" / "uploads"

all_texts = []

for pdf_file in PDF_DIR.glob("*.pdf"):
    print(f"📄 Loading {pdf_file.name}")
    pages = load_pdf_text(str(pdf_file))
    all_texts.extend(pages)

if not all_texts:
    raise ValueError("No text found in PDFs.")


# -------------------------
# Embedding model
# -------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda text: np.array(embed_model.encode(text))


# -------------------------
# Build & save index
# -------------------------
store = build_index_from_texts(all_texts, embed_fn)
store.save(str(ROOT_DIR / "vector_store"))

print("✅ Vector store built from PDFs and saved successfully.")
