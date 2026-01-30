import os
import google.generativeai as genai

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")

# Configure Gemini (official way)
genai.configure(api_key=API_KEY)

# Use currently supported public model
model = genai.GenerativeModel("gemini-1.5-flash")

def ask_gemini(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)

        if response and response.text:
            return response.text

        return "Sorry — I couldn’t generate a response."

    except Exception as e:
        print("Gemini error:", e)
        return "AI service temporarily unavailable. Please try again."
