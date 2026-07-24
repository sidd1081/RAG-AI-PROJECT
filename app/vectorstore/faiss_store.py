import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from app.config.settings import (
    FAISS_INDEX_TYPE,
    FAISS_HNSW_M,
    FAISS_HNSW_EF_CONSTRUCTION,
    FAISS_HNSW_EF_SEARCH,
)


class FAISSVectorStore:
    """
    FAISS index over normalized vectors (inner product == cosine similarity
    when vectors are L2-normalized at embedding time - see embedder.py).

    Two backends:
    - "flat" (IndexFlatIP): exact search, fine up to a few hundred thousand
      chunks, gets slow beyond that since it's brute-force.
    - "hnsw" (IndexHNSWFlat): approximate nearest neighbor via a navigable
      graph, scales to millions of chunks with sub-linear search time, at
      the cost of not guaranteeing the literal top-k (tunable via ef_search
      - higher ef_search trades speed for closer-to-exact recall).
    """

    def __init__(self, embedding_dim: int, index_type: str = FAISS_INDEX_TYPE):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = self._build_index(embedding_dim, index_type)
        self.metadata_store: List[Dict] = []

    @staticmethod
    def _build_index(embedding_dim: int, index_type: str):
        if index_type == "hnsw":
            index = faiss.IndexHNSWFlat(embedding_dim, FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = FAISS_HNSW_EF_CONSTRUCTION
            index.hnsw.efSearch = FAISS_HNSW_EF_SEARCH
            return index
        return faiss.IndexFlatIP(embedding_dim)

    def add_embeddings(self, embedded_chunks: List[Dict]):
        vectors = np.array(
            [chunk["embedding"] for chunk in embedded_chunks]
        ).astype("float32")

        self.index.add(vectors)
        self.metadata_store.extend(embedded_chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        query_vector = query_vector.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.metadata_store):
                item = dict(self.metadata_store[idx])
                item["vector_score"] = float(score)
                results.append(item)

        return results

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata_store, f)

    @classmethod
    def load(cls, directory: Path, embedding_dim: int) -> "FAISSVectorStore":
        store = cls.__new__(cls)
        store.embedding_dim = embedding_dim
        store.index = faiss.read_index(str(directory / "index.faiss"))
        store.index_type = FAISS_INDEX_TYPE
        with open(directory / "metadata.pkl", "rb") as f:
            store.metadata_store = pickle.load(f)
        return store
