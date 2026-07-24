import logging
import uuid
from typing import Callable, Dict, List

from app.config.settings import (
    CHUNK_TOKEN_SIZE,
    CHUNK_TOKEN_OVERLAP,
    MIN_CHUNK_TOKENS,
    EMBEDDING_MAX_TOKENS,
)
from app.preprocessing.sentence_splitter import split_sentences

logger = logging.getLogger(__name__)


def _hard_split_oversized(
    text: str, count_tokens: Callable[[str], int], max_tokens: int
) -> List[str]:
    """
    Safety net for anything that reaches here still over the embedding
    model's token limit - an oversized table, an oversized image caption,
    or a single sentence with no punctuation the sentence-splitter could
    break on. Packs whitespace-delimited words up to max_tokens per part.
    This is a blunt fallback (word-level, not semantic), only meant to run
    on the rare chunk that the normal sentence-aware packing couldn't keep
    under budget - it should almost never trigger on well-formed text.
    """
    words = text.split()
    if not words:
        return [text]

    parts: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for word in words:
        word_tokens = count_tokens(word)
        if current and current_tokens + word_tokens > max_tokens:
            parts.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += word_tokens

    if current:
        parts.append(" ".join(current))

    return parts if parts else [text]


def chunk_page(block: Dict, count_tokens: Callable[[str], int]) -> List[Dict]:
    """
    Token-aware semantic chunking.

    - Tables and images are never split; each becomes exactly one chunk,
      since splitting a table row or an image caption destroys its meaning.
    - Text blocks are packed sentence-by-sentence into a token budget
      (CHUNK_TOKEN_SIZE), never cutting a sentence in half, with the last
      few sentences of each chunk carried forward as overlap.
    """
    content_type = block.get("content_type", "text")

    if content_type in ("table", "image"):
        text = block["text"].strip()
        if not text:
            return []
        return [_build_chunk(block, text, content_type, chunk_index=0)]

    sentences = split_sentences(block["text"])
    if not sentences:
        return []

    chunks: List[Dict] = []
    current: List[str] = []
    current_tokens = 0
    chunk_index = 0

    def flush():
        nonlocal chunk_index
        if not current:
            return
        text = " ".join(current)
        if count_tokens(text) >= MIN_CHUNK_TOKENS:
            chunks.append(_build_chunk(block, text, "text", chunk_index))
            chunk_index += 1

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if current and current_tokens + sentence_tokens > CHUNK_TOKEN_SIZE:
            flush()

            overlap_sentences: List[str] = []
            overlap_tokens = 0
            for s in reversed(current):
                t = count_tokens(s)
                if overlap_tokens + t > CHUNK_TOKEN_OVERLAP:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += t

            current = overlap_sentences
            current_tokens = overlap_tokens

        current.append(sentence)
        current_tokens += sentence_tokens

    flush()
    return chunks


def _build_chunk(block: Dict, text: str, content_type: str, chunk_index: int) -> Dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "session_id": block["session_id"],
        "source_pdf": block["source_pdf"],
        "doc_hash": block["doc_hash"],
        "page_number": block["page_number"],
        "section_title": block.get("section_title"),
        "content_type": content_type,
        "chunk_index": chunk_index,
        "chunk_text": text,
        "image_path": block.get("image_path"),
        "source": block.get("source", "native"),
    }


def chunk_documents(blocks: List[Dict], count_tokens: Callable[[str], int]) -> List[Dict]:
    """
    Chunks every block, enforces the hard token-limit safety net on the
    result, then re-numbers chunk_index/total_chunks_in_doc per source
    document (chunk_page numbers chunks per-block, not per-doc).
    """
    all_chunks: List[Dict] = []
    for block in blocks:
        all_chunks.extend(chunk_page(block, count_tokens))

    all_chunks = _enforce_token_limit(all_chunks, count_tokens)

    by_doc: Dict[str, List[Dict]] = {}
    for c in all_chunks:
        by_doc.setdefault(c["source_pdf"], []).append(c)

    for doc_chunks in by_doc.values():
        total = len(doc_chunks)
        for i, c in enumerate(doc_chunks):
            c["chunk_index"] = i
            c["total_chunks_in_doc"] = total

    return all_chunks


def _enforce_token_limit(
    chunks: List[Dict], count_tokens: Callable[[str], int]
) -> List[Dict]:
    result: List[Dict] = []

    for chunk in chunks:
        if count_tokens(chunk["chunk_text"]) <= EMBEDDING_MAX_TOKENS:
            result.append(chunk)
            continue

        parts = _hard_split_oversized(chunk["chunk_text"], count_tokens, EMBEDDING_MAX_TOKENS)
        logger.warning(
            "Chunk (%s, page %s, %s) exceeded %d tokens and was force-split "
            "into %d parts to avoid silent embedding-model truncation.",
            chunk["source_pdf"], chunk["page_number"], chunk["content_type"],
            EMBEDDING_MAX_TOKENS, len(parts),
        )
        for part_text in parts:
            split_chunk = dict(chunk)
            split_chunk["chunk_id"] = str(uuid.uuid4())
            split_chunk["chunk_text"] = part_text
            result.append(split_chunk)

    return result
