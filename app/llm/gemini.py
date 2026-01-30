import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use stable supported model
model = genai.GenerativeModel("models/gemini-1.0-pro")

def ask_gemini(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text
