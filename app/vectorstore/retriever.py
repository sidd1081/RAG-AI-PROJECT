from typing import Dict, List

from app.config.settings import HYBRID_CANDIDATE_K, RRF_K, TOP_K_RETRIEVAL, USE_RERANKER
from app.embeddings.embedder import EmbeddingGenerator
from app.vectorstore.bm25_store import BM25Store
from app.vectorstore.fusion import reciprocal_rank_fusion
from app.vectorstore.reranker import Reranker


class HybridRetriever:
    """
    Hybrid retrieval: vector search (semantic) + BM25 (keyword), merged
    with Reciprocal Rank Fusion, then narrowed with a cross-encoder rerank.

    use_bm25/use_rerank can be overridden per-instance (see eval/run_eval.py)
    to compare retrieval strategies against the same underlying index.
    """

    def __init__(
        self,
        vector_store,
        bm25_store: BM25Store,
        top_k: int = TOP_K_RETRIEVAL,
        use_bm25: bool = True,
        use_rerank: bool = None,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.top_k = top_k
        self.use_bm25 = use_bm25
        self.embedder = EmbeddingGenerator()

        if use_rerank is None:
            use_rerank = USE_RERANKER
        self.reranker = Reranker() if use_rerank else None

    def retrieve(self, query: str) -> List[Dict]:
        query_embedding = self.embedder.embed_texts([query])[0]

        vector_results = self.vector_store.search(query_embedding, HYBRID_CANDIDATE_K)
        bm25_results = (
            self.bm25_store.search(query, HYBRID_CANDIDATE_K) if self.use_bm25 else []
        )

        fused = reciprocal_rank_fusion(vector_results, bm25_results, RRF_K)

        if self.reranker:
            return self.reranker.rerank(query, fused, self.top_k)

        return fused[: self.top_k]
