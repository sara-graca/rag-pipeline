"""All paths, model names and tunable settings, defined once for the whole project."""

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]  # the project folder (src/ is one level down)
DATA_DIR = ROOT / "data"

PDF_PATH = DATA_DIR / "rulebook.pdf"
PARSED_DIR = DATA_DIR / "parsed"
MD_PATH = PARSED_DIR / "rulebook_final.md"

PERSIST_DIR = DATA_DIR / "chroma_db_rulebook"

# --- Models ---
EMBEDDING_MODEL = "models/gemini-embedding-001"

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Retrieval ---
TOP_K = 5