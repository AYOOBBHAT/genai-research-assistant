from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import os

from app.core import state
from app.agent.rag_agent import build_agent
from app.ingest.pdf_loader import load_pdf_and_split
from app.vector.faiss_store import build_vector_store

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Reset index state when new PDF uploaded
    state.index_ready = False
    state.vector_store = None
    state.retriever = None
    state.agent = None

    return {
        "message": "PDF uploaded successfully. Index will be built on first query.",
        "filename": file.filename
    }

@router.post("/query")
async def query_rag(request: QueryRequest):
    try:
        # 🔥 Build index ONLY ONCE
        if not state.index_ready:
            print("📦 Building vector index (one-time)...")

            docs = load_pdf_and_split(UPLOAD_DIR)
            vector_store = build_vector_store(docs)

            retriever = vector_store.as_retriever(
                search_kwargs={"k": request.top_k}
            )

            agent = build_agent(retriever)

            state.vector_store = vector_store
            state.retriever = retriever
            state.agent = agent
            state.index_ready = True

            print("✅ Index built successfully")

        # ⚡ Fast path
        response = state.agent.invoke({"input": request.query})

        return {
            "answer": response["output"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
