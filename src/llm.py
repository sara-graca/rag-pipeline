"""The chat model used everywhere in the project (rewriting, answering, later routing and SQL)."""

from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLM_MODEL, LLM_TEMPERATURE

load_dotenv()  # reads GOOGLE_API_KEY from the .env file


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
#        temperature=LLM_TEMPERATURE,   # not used for flash-lite
        thinking_level="low",           # to have faster answers
    )

def text_of(message) -> str:   # newer models return list of parts
    """Return a model reply as plain text, whether it's a string or a list of parts."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))