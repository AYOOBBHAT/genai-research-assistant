from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rag.pipeline import rag_answer

router = APIRouter()

# These will be injected from main.py
retriever = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if retriever is None:
        raise HTTPException(
            status_code=500,
            detail="Vector store not loaded. Build and save index first."
        )

    return rag_answer(
        query=request.query,
        retriever=retriever,
        top_k=request.top_k
    )
