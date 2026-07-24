import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

elif not GROQ_API_KEY.startswith("gsk_"):
    raise ValueError("GROQ_API_KEY is not valid")

else:
    print("GROQ API key loaded successfully")
