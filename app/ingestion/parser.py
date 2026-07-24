import re
from typing import List, Dict
import pymupdf4llm

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


def extract_pages_markdown(pdf_path: str) -> List[Dict]:
    """
    Extracts each page as layout-aware Markdown (tables preserved as
    Markdown tables, headings preserved, multi-column reading order handled),
    then splits each page into text/table/heading blocks.
    """
    raw_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    blocks: List[Dict] = []
    for i, raw in enumerate(raw_pages):
        page_number = raw.get("metadata", {}).get("page", i + 1)
        text = raw.get("text", "")
        blocks.extend(_split_page_blocks(text, page_number))

    return blocks


def _split_page_blocks(markdown_text: str, page_number: int) -> List[Dict]:
    lines = markdown_text.splitlines()
    blocks: List[Dict] = []

    current_lines: List[str] = []
    current_type = "text"
    section_title = None

    def flush():
        text = "\n".join(current_lines).strip()
        if text:
            blocks.append({
                "page_number": page_number,
                "text": text,
                "content_type": current_type,
                "section_title": section_title,
            })
        current_lines.clear()

    for line in lines:
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush()
            current_type = "text"
            section_title = heading_match.group(2).strip()
            continue

        is_table_line = bool(_TABLE_LINE_RE.match(line))

        if is_table_line and current_type != "table":
            flush()
            current_type = "table"
        elif not is_table_line and line.strip() == "" and current_type == "table":
            flush()
            current_type = "text"

        current_lines.append(line)

    flush()
    return blocks
