import hashlib
from pathlib import Path


def load_pdf(pdf_path: str) -> Path:
    """
    Validates the PDF exists and returns its Path.
    Parsing (text + images) is done directly from this path by
    parser.py and image_extractor.py using PyMuPDF/pymupdf4llm.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    return pdf_path


def get_page_count(pdf_path: str) -> int:
    import fitz
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def compute_doc_hash(pdf_path: str) -> str:
    """
    Stable content hash used to dedupe re-uploads and namespace
    extracted images on disk.
    """
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]
