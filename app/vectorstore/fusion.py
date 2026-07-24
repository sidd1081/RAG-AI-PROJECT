from typing import Dict, List


def reciprocal_rank_fusion(
    vector_results: List[Dict], bm25_results: List[Dict], rrf_k: int
) -> List[Dict]:
    """
    Merges two ranked lists using Reciprocal Rank Fusion. Uses each list's
    rank order rather than raw scores, since vector similarity and BM25
    scores live on different, incomparable scales - this needs no score
    normalization tuning.
    """
    scores: Dict[str, float] = {}
    chunk_by_id: Dict[str, Dict] = {}

    for rank, item in enumerate(vector_results):
        chunk_id = item["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        chunk_by_id[chunk_id] = item

    for rank, item in enumerate(bm25_results):
        chunk_id = item["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        chunk_by_id.setdefault(chunk_id, item)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunk_by_id[cid] for cid in ranked_ids]
