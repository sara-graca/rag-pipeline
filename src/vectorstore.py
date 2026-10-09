"""Shared vector store setup, used for both indexing and querying.

The embedding model and storage location come from config, which guarantees that
chunks and questions are embedded with the same model.
"""

from functools import lru_cache

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import EMBEDDING_MODEL, PERSIST_DIR, TOP_K

load_dotenv()  # reads GOOGLE_API_KEY from the .env file


@lru_cache(maxsize=1)
def get_vector_db() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)


def retrieve(question: str, k: int = TOP_K):
    """Return the k chunks most similar to the question (the text-channel retriever)."""
    return get_vector_db().similarity_search(question, k=k)