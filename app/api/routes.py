from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi import UploadFile, File
from pathlib import Path
import shutil
import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.pdf_loader import load_pdf_text
from app.rag.pipeline import build_index_from_texts
from app.rag.vector_store import FaissVectorStore
from app.agent.rag_agent import build_agent
from app.config.settings import VECTOR_STORE_PATH
from app.config.settings import UPLOAD_DIR, VECTOR_STORE_PATH


router = APIRouter()

# Injected from main.py at startup
agent = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]] = []


@router.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    if agent is None:
        raise HTTPException(
            status_code=500,
            detail="Agent not initialized."
        )

    result = agent.invoke({
        "query": request.query,
        "rag_result": None,
        "final_answer": None,
        "confidence": None,
    })

    return {
        "answer": result["final_answer"],
        "confidence": result.get("confidence", 0.0),
        "sources": (
            result["rag_result"]["sources"]
            if result.get("rag_result") else []
        ),
    }
    
    
@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pages = load_pdf_text(str(file_path))
    if not pages:
        raise HTTPException(status_code=400, detail="No readable text found in PDF")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_fn = lambda text: np.array(embed_model.encode(text))

    store = build_index_from_texts(pages, embed_fn)
    store.save(str(VECTOR_STORE_PATH))

    from app.api import routes
    from app.rag.retriever import Retriever

    retriever = Retriever(embed_fn, store)
    routes.agent = build_agent(retriever)

    return {
        "message": "PDF uploaded and indexed successfully",
        "filename": file.filename,
        "pages_indexed": len(pages),
    }
