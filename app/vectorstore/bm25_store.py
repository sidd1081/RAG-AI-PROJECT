import pickle
import re
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    """
    Keyword-based retrieval over the same chunks indexed in FAISS.
    Catches exact matches (error codes, SKUs, proper nouns, numbers)
    that embedding similarity often misses.
    """

    def __init__(self):
        self.chunks: List[Dict] = []
        self.bm25: BM25Okapi = None

    def build(self, embedded_chunks: List[Dict]):
        self.chunks = embedded_chunks
        tokenized_corpus = [_tokenize(c["chunk_text"]) for c in embedded_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        if self.bm25 is None or not self.chunks:
            return []

        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked[:top_k]:
            if scores[idx] <= 0:
                continue
            item = dict(self.chunks[idx])
            item["bm25_score"] = float(scores[idx])
            results.append(item)

        return results

    def save(self, directory: Path):
        """
        Persists the underlying chunks, not the BM25Okapi object itself -
        rebuilding the index from chunks on load is cheap and avoids
        pickling issues with rank_bm25's internals.
        """
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "bm25_chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, directory: Path) -> "BM25Store":
        store = cls()
        with open(directory / "bm25_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        store.build(chunks)
        return store
