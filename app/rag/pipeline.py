import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from app.config.settings import (
    DESCRIBE_IMAGES,
    EMBEDDING_DIMENSION,
    session_image_dir,
    session_vector_dir,
)
from app.rag.manifest import write_manifest, manifest_is_compatible
from app.ingestion.pdf_loader import load_pdf, compute_doc_hash, get_page_count
from app.ingestion.parser import extract_pages_markdown
from app.ingestion.image_extractor import extract_images
from app.ingestion.image_describer import ImageDescriber
from app.ingestion.ocr import ocr_page, needs_ocr
from app.ingestion.metadata import attach_metadata
from app.preprocessing.cleaner import clean_text
from app.preprocessing.chunker import chunk_documents
from app.embeddings.embedder import EmbeddingGenerator
from app.vectorstore.faiss_store import FAISSVectorStore
from app.vectorstore.bm25_store import BM25Store

logger = logging.getLogger(__name__)


# =====================================================
# Index loading (session-scoped)
# =====================================================

def load_existing_index(session_id: str) -> Tuple[Optional[FAISSVectorStore], Optional[BM25Store]]:
    """
    Loads the FAISS + BM25 index for ONE session only. Every caller must
    pass a session_id - there is no "global" index. This is the isolation
    boundary between users/sessions.
    """
    directory = session_vector_dir(session_id)
    index_file = directory / "index.faiss"
    chunks_file = directory / "bm25_chunks.pkl"

    if not index_file.exists() or not chunks_file.exists():
        return None, None

    if not manifest_is_compatible(directory):
        logger.warning(
            "Index at %s was built with a different embedding model or "
            "schema version than the current settings. Refusing to load "
            "it to avoid corrupted similarity scores - re-upload documents "
            "to rebuild this session's index from scratch.",
            directory,
        )
        return None, None

    vector_store = FAISSVectorStore.load(directory, EMBEDDING_DIMENSION)
    bm25_store = BM25Store.load(directory)
    return vector_store, bm25_store


# =====================================================
# Ingestion
# =====================================================

def _extract_blocks_for_file(file_info: Dict, doc_hash: str, describer) -> Tuple[List[Dict], Dict]:
    """
    Returns (enriched_blocks, stats) for one file. stats tracks things the
    caller needs to know about but that shouldn't silently disappear:
    OCR pages used, and images whose captioning failed.
    """
    pdf_path = load_pdf(file_info["file_path"])

    text_blocks = extract_pages_markdown(pdf_path)

    native_char_count = defaultdict(int)
    for b in text_blocks:
        native_char_count[b["page_number"]] += len(b["text"])

    total_pages = get_page_count(pdf_path)
    ocr_pages_used = 0

    for page_number in range(1, total_pages + 1):
        if not needs_ocr(native_char_count.get(page_number, 0)):
            continue

        ocr_text = ocr_page(pdf_path, page_number)
        if needs_ocr(len(ocr_text)):
            continue  # OCR itself came back empty/too short - nothing to add

        text_blocks.append({
            "page_number": page_number,
            "text": ocr_text,
            "content_type": "text",
            "section_title": None,
            "source": "ocr",
        })
        ocr_pages_used += 1

    image_output_dir = session_image_dir(file_info["session_id"])
    image_blocks = extract_images(pdf_path, doc_hash, image_output_dir)

    failed_image_count = 0
    if describer:
        for img in image_blocks:
            img["text"] = describer.describe(img["image_path"])
            if not img["text"]:
                failed_image_count += 1
    else:
        for img in image_blocks:
            img["text"] = ""

    raw_blocks = text_blocks + [b for b in image_blocks if b["text"]]

    enriched = attach_metadata(
        blocks=raw_blocks,
        session_id=file_info["session_id"],
        source_pdf=file_info["file_name"],
        doc_hash=doc_hash,
    )

    stats = {
        "ocr_pages_used": ocr_pages_used,
        "images_found": len(image_blocks),
        "images_failed": failed_image_count,
    }
    return enriched, stats


def ingest_documents(uploaded_documents: List[Dict], session_id: str):
    """
    Full ingestion pipeline, scoped entirely to one session:

    - Loads that session's existing index from disk, if compatible.
    - Skips re-embedding any uploaded file whose doc_hash is already
      indexed for THIS session.
    - New files: parse -> OCR fallback -> images -> clean -> chunk -> embed.
    - Merges into the session's index and saves it back to disk.

    Returns (all_chunks, vector_store, bm25_store, stats).
    """
    embedder = EmbeddingGenerator()
    describer = ImageDescriber() if DESCRIBE_IMAGES else None

    existing_vector_store, existing_bm25_store = load_existing_index(session_id)
    existing_hashes = set()
    if existing_bm25_store:
        existing_hashes = {
            c["metadata"]["doc_hash"] for c in existing_bm25_store.chunks
        }

    new_blocks: List[Dict] = []
    total_stats = {"ocr_pages_used": 0, "images_found": 0, "images_failed": 0, "skipped_duplicates": 0}

    for file_info in uploaded_documents:
        pdf_path = load_pdf(file_info["file_path"])
        doc_hash = compute_doc_hash(pdf_path)

        if doc_hash in existing_hashes:
            total_stats["skipped_duplicates"] += 1
            continue

        blocks, stats = _extract_blocks_for_file(file_info, doc_hash, describer)
        new_blocks.extend(blocks)
        for key in ("ocr_pages_used", "images_found", "images_failed"):
            total_stats[key] += stats[key]

    if not new_blocks:
        if existing_vector_store:
            return existing_bm25_store.chunks, existing_vector_store, existing_bm25_store, total_stats
        return [], None, None, total_stats

    for block in new_blocks:
        if block["content_type"] != "image":
            block["text"] = clean_text(block["text"])

    new_chunks = chunk_documents(new_blocks, count_tokens=embedder.count_tokens)
    new_embedded_chunks = embedder.embed_chunks(new_chunks)

    if existing_vector_store and existing_bm25_store:
        vector_store = existing_vector_store
        vector_store.add_embeddings(new_embedded_chunks)
        all_chunks = existing_bm25_store.chunks + new_embedded_chunks
    else:
        vector_dim = new_embedded_chunks[0]["embedding"].shape[0]
        vector_store = FAISSVectorStore(vector_dim)
        vector_store.add_embeddings(new_embedded_chunks)
        all_chunks = new_embedded_chunks

    bm25_store = BM25Store()
    bm25_store.build(all_chunks)

    directory = session_vector_dir(session_id)
    vector_store.save(directory)
    bm25_store.save(directory)
    write_manifest(directory)

    return all_chunks, vector_store, bm25_store, total_stats
