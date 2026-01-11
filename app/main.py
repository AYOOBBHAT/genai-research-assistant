import logging
import numpy as np
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from app.api import routes
from app.agent.rag_agent import build_agent
from app.config.settings import VECTOR_STORE_PATH
from app.rag.vector_store import FaissVectorStore
from app.rag.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GenAI Research Assistant",
    version="1.0.0"
)


@app.on_event("startup")
def startup_event():
    logger.info(f"📂 Loading vector store from: {VECTOR_STORE_PATH}")

    try:
        # Embeddings
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        embed_fn = lambda text: np.array(embed_model.encode(text))

        # Load vector store
        store = FaissVectorStore.load(str(VECTOR_STORE_PATH))
        retriever = Retriever(embed_fn, store)

        # Build agent
        routes.agent = build_agent(retriever)

        logger.info("✅ Agent initialized successfully")

    except Exception as e:
        routes.agent = None
        logger.error(f"❌ Failed to initialize agent: {e}")


app.include_router(routes.router)


@app.get("/")
def health():
    return {"status": "ok"}
