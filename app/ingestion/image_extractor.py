from pathlib import Path
from typing import List, Dict
import fitz

from app.config.settings import MIN_IMAGE_DIMENSION_PX


def extract_images(pdf_path: str, doc_hash: str, output_dir: Path) -> List[Dict]:
    """
    Extracts embedded images from a PDF, saves them to disk, and returns
    one block per image with its page number and file path. Tiny images
    (icons/logos) below MIN_IMAGE_DIMENSION_PX are skipped.
    """
    doc = fitz.open(str(pdf_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    images: List[Dict] = []

    for page_index in range(len(doc)):
        page = doc[page_index]

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            if width < MIN_IMAGE_DIMENSION_PX or height < MIN_IMAGE_DIMENSION_PX:
                continue

            ext = base_image.get("ext", "png")
            filename = f"{doc_hash}_p{page_index + 1}_{img_index}.{ext}"
            image_path = output_dir / filename

            with open(image_path, "wb") as f:
                f.write(base_image["image"])

            images.append({
                "page_number": page_index + 1,
                "image_path": str(image_path),
                "content_type": "image",
            })

    doc.close()
    return images
