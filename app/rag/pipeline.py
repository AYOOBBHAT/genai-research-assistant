"""
RAG pipeline:
- Build index from raw texts
- Retrieve relevant chunks
- Generate grounded answers using retrieved context
"""

from typing import List, Dict, Any, Callable, Optional
import numpy as np



def build_index_from_texts(
    texts: List[str],
    embed_fn: Callable[[str], np.ndarray],
    store=None,
    chunk_size: int = 500,
    overlap: int = 50,
):
    """
    Takes raw texts → chunks → embeddings → FAISS store
    Returns a populated FaissVectorStore
    """
    from app.rag.chunker import chunk_text
    from app.rag.vector_store import FaissVectorStore

    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for source_idx, text in enumerate(texts):
        text_chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )
        for chunk_id, chunk in enumerate(text_chunks):
            chunks.append(chunk)
            metadatas.append({
                "source": f"doc_{source_idx}",
                "chunk_id": chunk_id,
                "text": chunk
            })

    if not chunks:
        raise ValueError("No chunks created from input texts.")

    embeddings = [embed_fn(chunk) for chunk in chunks]

    if store is None:
        store = FaissVectorStore()   

    store.add_texts(embeddings, metadatas)
    return store



def default_hf_generator(model_name: str = "gpt2"):
    """
    Lightweight text generator for local testing.
    CPU-only, no accelerate, no device_map.
    """
    from transformers import pipeline

    generator = pipeline(
        "text-generation",
        model=model_name,
        device=-1   
    )

    def generate(prompt: str, max_new_tokens: int = 50) -> str:
        output = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=50256
        )
        return output[0]["generated_text"]

    return generate



def rag_answer(
    query: str,
    retriever,
    llm_generate: Optional[Callable[[str], str]] = None,
    top_k: int = 4,
) -> Dict[str, Any]:
    """
    Full RAG step:
    query → retrieve → prompt → generate → structured output
    """
    if llm_generate is None:
        llm_generate = default_hf_generator()

    results = retriever.retrieve(query, k=top_k)

    if not results:
        return {
            "answer": "No relevant context found.",
            "sources": [],
            "confidence": 0.0
        }

    
    context = "\n\n".join(r["text"] for r in results)


    prompt = f"""
You are a factual assistant.
Answer ONLY using the context below.
If the answer is not present, reply exactly: "Not in context."

Context:
{context}

Question:
{query}

Answer:
"""

    
    raw = llm_generate(prompt)
    raw_text = raw.strip()

   
    if "Answer:" in raw_text:
        raw_text = raw_text.split("Answer:", 1)[1]

    
    stop_tokens = ["\nQuestion:", "\n\nQuestion:"]
    for token in stop_tokens:
        if token in raw_text:
            raw_text = raw_text.split(token, 1)[0]

    raw_answer = raw_text.strip()


    sources = [
        {
            "source": r["source"],
            "chunk_id": r["chunk_id"],
            "score": r["score"],
        }
        for r in results
    ]

    # Confidence heuristic (lower distance → higher confidence)
    top_score = results[0]["score"]
    confidence = float(1.0 / (1.0 + top_score))
    CONFIDENCE_THERSHOLD=0.75
    if confidence<CONFIDENCE_THERSHOLD:
        return{
            "answer":"i am not confident enougto answer",
            "sources":[],
            "confidence":confidence,
            
        }

    return {
        "answer": raw_answer,
        "sources": sources,
        "confidence": confidence,
    }
