from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import numpy as np
import logging

from sentence_transformers import SentenceTransformer

from app.core import state
from app.rag.pipeline import build_index_from_texts, rag_answer
from app.rag.vector_store import FaissVectorStore
from app.rag.retriever import Retriever
from app.rag.pdf_loader import load_pdf_text


router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = "data"
VECTOR_STORE_DIR = "vector_store"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda text: np.array(embed_model.encode(text))


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float



@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Reset state
    state.index_ready = False
    state.vector_store = None
    state.retriever = None

    logger.info(f"📄 Uploaded {file.filename}")

    return {
        "message": "PDF uploaded successfully. Index will be built on first query.",
        "filename": file.filename
    }



@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not state.index_ready:
        logger.info("📦 Building vector index (lazy)…")

        pdfs = [
            os.path.join(UPLOAD_DIR, f)
            for f in os.listdir(UPLOAD_DIR)
            if f.endswith(".pdf")
        ]

        if not pdfs:
            raise HTTPException(400, "No PDF uploaded yet")

        texts = []
        for pdf in pdfs:
            texts.extend(load_pdf_text(pdf))

        if not texts:
            raise HTTPException(400, "No text extracted from PDF")

        store = build_index_from_texts(texts, embed_fn)
        store.save(VECTOR_STORE_DIR)

        state.vector_store = store
        state.retriever = Retriever(store)
        state.index_ready = True

        logger.info("Index built")

    return rag_answer(
        query=request.query,
        retriever=state.retriever,
        top_k=request.top_k
    )
