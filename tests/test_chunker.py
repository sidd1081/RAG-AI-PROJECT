from app.preprocessing.chunker import chunk_page, chunk_documents, _hard_split_oversized


def _count_tokens(text):
    return len(text.split())


def _text_block(text, page=1, doc="doc.pdf"):
    return {
        "session_id": "s1",
        "source_pdf": doc,
        "doc_hash": "h1",
        "page_number": page,
        "text": text,
        "content_type": "text",
        "section_title": None,
        "image_path": None,
    }


def _table_block(text, page=1, doc="doc.pdf"):
    block = _text_block(text, page, doc)
    block["content_type"] = "table"
    return block


def test_table_block_is_never_split_by_chunk_page():
    long_table = "| a | b |\n" * 500
    block = _table_block(long_table)
    chunks = chunk_page(block, _count_tokens)
    assert len(chunks) == 1
    assert chunks[0]["content_type"] == "table"


def test_text_block_is_packed_into_multiple_chunks_when_over_budget():
    sentences = " ".join(f"This is sentence number {i}." for i in range(200))
    block = _text_block(sentences)
    chunks = chunk_page(block, _count_tokens)
    assert len(chunks) > 1
    for c in chunks:
        assert c["content_type"] == "text"


def test_hard_split_keeps_every_part_under_the_cap():
    text = " ".join(f"word{i}" for i in range(1000))
    parts = _hard_split_oversized(text, _count_tokens, max_tokens=50)
    assert len(parts) > 1
    for part in parts:
        assert _count_tokens(part) <= 50


def test_chunk_documents_enforces_token_cap_even_on_oversized_table():
    huge_table = " ".join(f"cell{i}" for i in range(2000))
    block = _table_block(huge_table)
    chunks = chunk_documents([block], _count_tokens)
    from app.config.settings import EMBEDDING_MAX_TOKENS
    for c in chunks:
        assert _count_tokens(c["chunk_text"]) <= EMBEDDING_MAX_TOKENS


def test_chunk_index_and_total_are_numbered_per_document():
    block = _text_block(
        "This is a reasonably long test sentence for numbering purposes. "
        "It has enough words across both sentences to clear the minimum chunk size."
    )
    chunks = chunk_documents([block], _count_tokens)
    assert len(chunks) > 0
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["total_chunks_in_doc"] == len(chunks)
