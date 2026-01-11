import logging
from fastapi import FastAPI

from sentence_transformers import SentenceTransformer
import numpy as np

from app.api.routes import router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


logger.info("🔹 Loading embedding model (once)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda text: np.array(embed_model.encode(text))
logger.info("✅ Embedding model loaded")


app = FastAPI(
    title="GenAI Research Assistant",
    version="1.0.0"
)


app.state.embed_fn = embed_fn
app.state.vector_store = None
app.state.retriever = None
app.state.agent = None


app.include_router(router)


@app.get("/")
def health():
    return {"status": "ok"}
