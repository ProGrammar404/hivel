import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in .env file.")

# Max tokens per single LLM call (as per assignment constraint)
MAX_TOKENS_PER_CALL = 16000

# Gemini model configuration
MODEL_NAME = "gemini-2.0-flash"


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Create and return a configured Gemini LLM instance."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        max_output_tokens=4096,
    )
