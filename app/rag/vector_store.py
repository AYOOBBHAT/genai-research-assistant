from typing import List, Dict, Any
import numpy as np
import os
import pickle
import faiss


class FaissVectorStore:
    def __init__(self):
        self.index = None
        self.dim = None
        self.id_to_meta: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
            self.dim = dim

    def add_texts(self, embeddings: List[np.ndarray], metadatas: List[Dict[str, Any]]):
        if not embeddings:
            return

        dim = embeddings[0].shape[0]
        self._ensure_index(dim)

        vectors = np.vstack(embeddings).astype("float32")
        self.index.add(vectors)

        for meta in metadatas:
            self.id_to_meta[self.next_id] = meta
            self.next_id += 1

    def search(self, query_embedding: np.ndarray, k: int = 4):
        if self.index is None:
            return [], []

        q = query_embedding.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(q, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            meta = dict(self.id_to_meta[idx])
            meta["score"] = float(dist)
            results.append(meta)

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump(self.id_to_meta, f)

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            self.id_to_meta = pickle.load(f)
