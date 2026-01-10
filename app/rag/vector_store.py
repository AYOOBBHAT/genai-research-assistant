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

    # -------------------------------------------------
    # Internal: ensure FAISS index exists
    # -------------------------------------------------
    def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
            self.dim = dim

    # -------------------------------------------------
    # Add embeddings + metadata
    # -------------------------------------------------
    def add_texts(
        self,
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]],
    ):
        if not embeddings:
            return

        dim = embeddings[0].shape[0]
        self._ensure_index(dim)

        vectors = np.vstack(embeddings).astype("float32")
        self.index.add(vectors)

        for meta in metadatas:
            self.id_to_meta[self.next_id] = meta
            self.next_id += 1

    # -------------------------------------------------
    # Search
    # -------------------------------------------------
    def search(self, query_embedding: np.ndarray, k: int = 4):
        if self.index is None:
            return []

        q = query_embedding.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(q, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            meta = dict(self.id_to_meta.get(idx, {}))
            meta["score"] = float(dist)
            results.append(meta)

        return results

    # -------------------------------------------------
    # Save (NEW SCHEMA)
    # -------------------------------------------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump(
                {
                    "id_to_meta": self.id_to_meta,
                    "next_id": self.next_id,
                    "dim": self.dim,
                },
                f,
            )

    # -------------------------------------------------
    # Load (BACKWARD + FORWARD COMPATIBLE)
    # -------------------------------------------------
    @classmethod
    def load(cls, path: str):
        store = cls()

        # Load FAISS index
        store.index = faiss.read_index(os.path.join(path, "index.faiss"))

        # Load metadata (handle multiple schema versions)
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            data = pickle.load(f)

            # ---- NEW SCHEMA ----
            if isinstance(data, dict) and "id_to_meta" in data:
                store.id_to_meta = data["id_to_meta"]
                store.next_id = data.get("next_id", len(store.id_to_meta))
                store.dim = data.get("dim")

            # ---- OLD SCHEMA (dict[int, meta]) ----
            elif isinstance(data, dict):
                store.id_to_meta = data
                store.next_id = len(data)
                store.dim = None

            else:
                raise ValueError("Unsupported meta.pkl format")

        return store
