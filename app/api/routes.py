from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import numpy as np

from sentence_transformers import SentenceTransformer

from app.core import state
from app.agent.rag_agent import build_agent
from app.rag.pdf_loader import load_pdf_text
from app.rag.pipeline import build_index_from_texts
from app.rag.retriever import Retriever



UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4



embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_fn = lambda text: np.array(embed_model.encode(text))



@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    
    state.index_ready = False
    state.vector_store = None
    state.retriever = None
    state.agent = None

    return {
        "message": "PDF uploaded successfully. Index will be built on first query.",
        "filename": file.filename,
    }



@router.post("/query")
async def query_rag(request: QueryRequest):
    try:
        
        if not state.index_ready:
            print("📦 Building vector index (one-time)...")

            
            texts = []
            for filename in os.listdir(UPLOAD_DIR):
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(UPLOAD_DIR, filename)
                    texts.extend(load_pdf_text(pdf_path))

            if not texts:
                raise HTTPException(
                    status_code=400,
                    detail="No PDF documents found. Upload a PDF first.",
                )

            
            vector_store = build_index_from_texts(texts, embed_fn)

            
            retriever = Retriever(vector_store)

            
            agent = build_agent(retriever)

            
            state.vector_store = vector_store
            state.retriever = retriever
            state.agent = agent
            state.index_ready = True

            print("✅ Index built successfully")


        response = state.agent.invoke({"input": request.query})

        return {
            "answer": response["output"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
