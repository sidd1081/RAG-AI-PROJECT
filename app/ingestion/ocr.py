import io
import fitz
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image

from app.config.settings import OCR_FALLBACK_MIN_CHARS


def needs_ocr(native_char_count: int) -> bool:
    """
    A page needs OCR if its NATIVE extracted text is under the threshold -
    this must be checked by total character count per page, not by
    "did pymupdf4llm return zero blocks for this page". A mostly-scanned
    page can still yield one small text block (e.g. a page number or a
    stray header), which would otherwise incorrectly skip OCR.
    """
    return native_char_count < OCR_FALLBACK_MIN_CHARS


def ocr_page(pdf_path: str, page_number: int, dpi: int = 200) -> str:
    """
    Renders one page of the PDF to an image and runs Tesseract OCR on it.
    Used as a fallback for scanned pages that have no extractable text
    layer (pymupdf4llm returns nothing for these).

    Requires the tesseract-ocr binary installed at the OS level
    (e.g. `apt install tesseract-ocr`) - pytesseract is just a wrapper
    around that binary, not a standalone OCR engine.
    """
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_number - 1]
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""
