from app.ingestion.ocr import needs_ocr
from app.config.settings import OCR_FALLBACK_MIN_CHARS


def test_empty_page_needs_ocr():
    assert needs_ocr(0) is True


def test_page_with_a_few_stray_characters_still_needs_ocr():
    # This is the exact bug being guarded against: a page that yields one
    # tiny native text block (e.g. a page number) must still trigger OCR,
    # not be treated as "already has text".
    assert needs_ocr(5) is True


def test_page_with_enough_native_text_does_not_need_ocr():
    assert needs_ocr(OCR_FALLBACK_MIN_CHARS + 100) is False


def test_boundary_is_exclusive():
    assert needs_ocr(OCR_FALLBACK_MIN_CHARS) is False
    assert needs_ocr(OCR_FALLBACK_MIN_CHARS - 1) is True
