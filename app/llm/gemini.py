import os
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

client = genai.Client(api_key=API_KEY)

def ask_gemini(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-1.0-pro",
            contents=prompt,
        )

        return response.text or "Sorry — I couldn’t generate a response."

    except Exception as e:
        print("Gemini error:", e)
        return "AI service temporarily unavailable. Please try again."
