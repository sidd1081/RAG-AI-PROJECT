import base64
import logging
from groq import Groq

from app.config.env import GROQ_API_KEY
from app.config.settings import VISION_MODEL_NAME

logger = logging.getLogger(__name__)

_DESCRIBE_PROMPT = (
    "Describe this image factually for a document search index. "
    "Include any visible text, numbers, labels, chart type, axis values, "
    "or table contents exactly as shown. Do not speculate beyond what is visible."
)

_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


class ImageDescriber:
    """
    Generates a factual, retrieval-friendly caption for an image using a
    Groq-hosted vision model. Retries once on failure; if both attempts
    fail, logs a warning and returns "" - the caller (pipeline.py) is
    responsible for tracking and surfacing these failures to the user
    rather than letting the image vanish from the index silently.
    """

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def describe(self, image_path: str) -> str:
        for attempt in range(2):
            try:
                return self._call_vision_model(image_path)
            except Exception as exc:
                if attempt == 0:
                    continue
                logger.warning(
                    "Image captioning failed for %s after retry: %s", image_path, exc
                )
                return ""
        return ""

    def _call_vision_model(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        ext = image_path.rsplit(".", 1)[-1].lower()
        mime = _MIME_BY_EXT.get(ext, "image/png")

        response = self.client.chat.completions.create(
            model=VISION_MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _DESCRIBE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            }],
            temperature=0.0,
            max_tokens=300,
        )
        text = response.choices[0].message.content.strip()
        if not text:
            raise ValueError("Vision model returned an empty caption")
        return text
