from typing import Callable, List, Dict, Any
import numpy as np
from app.rag.vector_store import FaissVectorStore

class Retriever:
    def __init__(self, embed_fn: Callable[[str], np.ndarray], store):
        self.embed_fn = embed_fn
        self.store = store

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        query_embedding = self.embed_fn(query)
        results = self.store.search(query_embedding, k)
        return results
