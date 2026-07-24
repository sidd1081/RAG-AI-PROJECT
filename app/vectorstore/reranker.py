import streamlit as st
from sentence_transformers import CrossEncoder

from app.config.settings import RERANK_MODEL_NAME


@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANK_MODEL_NAME)


class Reranker:
    """
    Cross-encoder reranking: scores (query, chunk) pairs jointly, which is
    substantially more accurate than comparing precomputed embedding
    vectors independently. Run only on the small candidate set returned
    by hybrid retrieval, since cross-encoders are too slow to run over an
    entire corpus.
    """

    def __init__(self):
        self.model = load_reranker()

    def rerank(self, query: str, candidates: list, top_k: int) -> list:
        if not candidates:
            return []

        pairs = [(query, c["chunk_text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]
