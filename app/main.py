import logging
import numpy as np
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from app.api import routes
from app.config.settings import VECTOR_STORE_PATH
from app.rag.vector_store import FaissVectorStore
from app.rag.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GenAI Research Assistant",
    version="1.0.0"
)

# -------------------------
# Startup event (CORRECT)
# -------------------------
@app.on_event("startup")
def startup_event():
    logger.info(f"📂 Loading vector store from: {VECTOR_STORE_PATH}")

    try:
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        embed_fn = lambda text: np.array(embed_model.encode(text))

        store = FaissVectorStore.load(str(VECTOR_STORE_PATH))
        routes.retriever = Retriever(embed_fn, store)

        logger.info("✅ Vector store loaded successfully")

    except Exception as e:
        routes.retriever = None
        logger.error(f"❌ Failed to load vector store: {e}")


# -------------------------
# Routes
# -------------------------
app.include_router(routes.router)


@app.get("/")
def health():
    return {"status": "ok"}
