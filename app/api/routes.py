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



UPLOAD_DIR = "data"
VECTOR_STORE_DIR = "vector_store"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
router = APIRouter()



logger.info("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda text: np.array(embed_model.encode(text))
logger.info(" Embedding model loaded")



class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


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

    logger.info(f" PDF uploaded: {file.filename}")
    logger.info(" Building vector index...")

    try:
        texts = load_pdf_text(file_path)
        if not texts:
            raise ValueError("No text extracted from PDF")

    
        vector_store = build_index_from_texts(texts, embed_fn)

        
        vector_store.save(VECTOR_STORE_DIR)

    
        retriever = Retriever(vector_store)

        
        state.vector_store = vector_store
        state.retriever = retriever
        state.index_ready = True

        logger.info(" Vector index built and ready")

        return {
            "message": "PDF uploaded and indexed successfully",
            "filename": file.filename,
            "chunks": len(texts)
        }

    except Exception as e:
        logger.exception(" Failed to build index")
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not state.index_ready or state.retriever is None:
        raise HTTPException(
            status_code=400,
            detail="Upload and index a PDF first"
        )

    try:
        result = rag_answer(
            query=request.query,
            retriever=state.retriever,
            top_k=request.top_k
        )
        return result

    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))
