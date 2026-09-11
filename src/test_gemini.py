import os
from dotenv import load_dotenv
from google import genai

load_dotenv()  # must be called in every module that reads os.environ directly

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found - check your .env file is in the project root")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Reply with exactly one word: pong",
)

print("Response:", response.text)
print("\nIf you see 'pong' above, the API key and SDK are working correctly.")