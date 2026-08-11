import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Groq endpoint configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY is not set in the environment variables.")

# Initialize global AsyncOpenAI client configured for Groq
llm_client = AsyncOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY
)

async def get_llm_client() -> AsyncOpenAI:
    """Returns the initialized AsyncOpenAI client instance."""
    return llm_client