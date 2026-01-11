import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from pydantic import BaseModel

import numpy as np

from app.rag.pdf_loader import load_pdf_text
from app.rag.pipeline import build_index_from_texts, rag_answer
from app.rag.retriever import Retriever
from app.agent.rag_agent import build_agent
from app.config.settings import UPLOAD_DIR


logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float


@router.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = UPLOAD_DIR / file.filename

    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    
    request.app.state.vector_store = None
    request.app.state.retriever = None
    request.app.state.agent = None

    logger.info(f"📄 PDF uploaded: {file.filename}")

    return {
        "message": "PDF uploaded successfully. Index will be built on first query.",
        "filename": file.filename,
    }



@router.post("/query", response_model=QueryResponse)
def query_rag(request: Request, payload: QueryRequest):
    embed_fn = request.app.state.embed_fn

   
    if request.app.state.retriever is None:
        logger.info("⚙️ Building vector index lazily...")

        pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
        if not pdf_files:
            raise HTTPException(
                status_code=400,
                detail="No PDFs uploaded. Upload a document first."
            )

        all_texts = []
        for pdf in pdf_files:
            pages = load_pdf_text(str(pdf))
            all_texts.extend(pages)

        if not all_texts:
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDFs contain no readable text."
            )

        
        store = build_index_from_texts(all_texts, embed_fn)
        retriever = Retriever(embed_fn, store)
        agent = build_agent(retriever)

        
        request.app.state.vector_store = store
        request.app.state.retriever = retriever
        request.app.state.agent = agent

        logger.info("✅ Vector index built and agent initialized")

    
    result = rag_answer(
        query=payload.query,
        retriever=request.app.state.retriever,
        top_k=payload.top_k,
    )

    return result
