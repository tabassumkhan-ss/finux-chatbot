import os
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

def ask_gemini(prompt: str) -> str:
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return resp.text or "Sorry — I could not generate a reply."

    except Exception as e:
        print("Gemini error:", e)
        return "Sorry — AI service temporarily unavailable."
