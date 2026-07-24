from groq import Groq
from app.config.env import GROQ_API_KEY
from app.config.settings import LLM_MODEL_NAME

class AnswerGenerator:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. "
                        "Answer ONLY using the provided context. "
                        "If the answer is not present, say: "
                        "'Answer not found in the provided document.'"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=256
        )

        return response.choices[0].message.content.strip()
