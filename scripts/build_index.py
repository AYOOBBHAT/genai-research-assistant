
import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.pipeline import build_index_from_texts
from app.rag.vector_store import FaissVectorStore

# -------------------------
# Load documents (example)
# -------------------------
texts = [
    "AI helps doctors diagnose diseases.",
    "Machine learning is used in medical imaging.",
]

# -------------------------
# Embedding model
# -------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda text: np.array(embed_model.encode(text))

# -------------------------
# Build & save index
# -------------------------
store = build_index_from_texts(texts, embed_fn)
store.save("vector_store")

print("✅ Vector store built and saved successfully.")
