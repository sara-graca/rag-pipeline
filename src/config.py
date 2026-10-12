"""All paths, model names and tunable settings, defined once for the whole project."""

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]  # the project folder (src/ is one level down)
DATA_DIR = ROOT / "data"

PDF_PATH = DATA_DIR / "raw/Core Rules.pdf"
PARSED_DIR = DATA_DIR / "parsed"
MD_PATH = PARSED_DIR / "rulebook_final.md"

PERSIST_DIR = DATA_DIR / "chroma_db_rulebook"

# --- Models ---
EMBEDDING_MODEL = "models/gemini-embedding-001"

# --- Chatbot ---
LLM_MODEL = "gemini-3.5-flash-lite"
LLM_TEMPERATURE = 0              # 0 = deterministic answers
MAX_HISTORY_TURNS = 5            # how many past question/answer pairs the bot remembers

# --- SQL ---
SQLITE_PATH = DATA_DIR / "raw/wahadb.sqlite"
SQL_DICTIONARY_PATH = DATA_DIR / "raw/SQL_AGENT_DATA_DICTIONARY.md"
SQL_MAX_ROWS = 50        # rows given to the LLM at most
SQL_MAX_ATTEMPTS = 2     # first try + one correction

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Retrieval ---
TOP_K = 5