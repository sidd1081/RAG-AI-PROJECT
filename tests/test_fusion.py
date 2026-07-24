from app.vectorstore.fusion import reciprocal_rank_fusion


def _chunk(chunk_id, text="text"):
    return {"chunk_id": chunk_id, "chunk_text": text}


def test_item_ranked_first_in_both_lists_wins():
    vector_results = [_chunk("a"), _chunk("b"), _chunk("c")]
    bm25_results = [_chunk("a"), _chunk("c"), _chunk("b")]

    fused = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=60)

    assert fused[0]["chunk_id"] == "a"


def test_item_only_in_one_list_still_included():
    vector_results = [_chunk("a"), _chunk("b")]
    bm25_results = [_chunk("c")]

    fused = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=60)
    fused_ids = {c["chunk_id"] for c in fused}

    assert fused_ids == {"a", "b", "c"}


def test_empty_bm25_list_falls_back_to_vector_order():
    vector_results = [_chunk("a"), _chunk("b"), _chunk("c")]

    fused = reciprocal_rank_fusion(vector_results, [], rrf_k=60)

    assert [c["chunk_id"] for c in fused] == ["a", "b", "c"]


def test_no_duplicate_entries_for_item_in_both_lists():
    vector_results = [_chunk("a")]
    bm25_results = [_chunk("a")]

    fused = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=60)

    assert len(fused) == 1
