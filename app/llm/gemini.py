import os
from google import genai
import traceback

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("WARNING: GEMINI_API_KEY missing")

client = genai.Client(api_key=API_KEY)


def ask_gemini(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt,
        )

        if hasattr(response, "text") and response.text:
            return response.text

        return "Sorry — I couldn’t generate a response."

    except Exception:
        traceback.print_exc()
        return "AI service temporarily unavailable. Please try again."
