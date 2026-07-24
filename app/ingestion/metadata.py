from typing import List, Dict


def attach_metadata(
    blocks: List[Dict],
    session_id: str,
    source_pdf: str,
    doc_hash: str,
) -> List[Dict]:
    """
    Attaches document-level metadata to every extracted block
    (text, table, or image) before chunking.
    """
    enriched = []

    for block in blocks:
        enriched.append({
            "session_id": session_id,
            "source_pdf": source_pdf,
            "doc_hash": doc_hash,
            "page_number": block["page_number"],
            "text": block.get("text", ""),
            "content_type": block.get("content_type", "text"),
            "section_title": block.get("section_title"),
            "image_path": block.get("image_path"),
            "source": block.get("source", "native"),
        })

    return enriched
