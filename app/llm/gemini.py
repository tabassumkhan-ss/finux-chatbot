import os
import traceback
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("WARNING: GEMINI_API_KEY missing")

client = genai.Client(api_key=API_KEY)


def ask_gemini(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model="models/gemini-pro",
            contents=prompt,
        )

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        return "Sorry — I couldn’t generate a response."

    except Exception:
        traceback.print_exc()
        return "AI service temporarily unavailable. Please try again."
