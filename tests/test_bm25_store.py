from app.vectorstore.bm25_store import BM25Store


def _chunk(chunk_id, text, doc_hash="h1"):
    return {
        "chunk_id": chunk_id,
        "chunk_text": text,
        "metadata": {"doc_hash": doc_hash, "source_pdf": "doc.pdf"},
    }


def test_search_returns_relevant_chunk():
    store = BM25Store()
    store.build([
        _chunk("1", "the invoice total is five hundred dollars"),
        _chunk("2", "the weather today is sunny and warm"),
        _chunk("3", "please review the quarterly budget report"),
    ])

    results = store.search("invoice total", top_k=5)

    assert results
    assert results[0]["chunk_id"] == "1"


def test_save_and_load_roundtrip(tmp_path):
    store = BM25Store()
    chunks = [
        _chunk("1", "apples and oranges"),
        _chunk("2", "bananas and grapes"),
        _chunk("3", "a completely unrelated sentence about cars"),
    ]
    store.build(chunks)
    store.save(tmp_path)

    loaded = BM25Store.load(tmp_path)
    results = loaded.search("apples", top_k=5)

    assert results
    assert results[0]["chunk_id"] == "1"


def test_search_on_empty_store_returns_empty_list():
    store = BM25Store()
    assert store.search("anything", top_k=5) == []
