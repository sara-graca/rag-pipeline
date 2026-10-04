import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from chunker import create_chunks

load_dotenv()  # reads GOOGLE_API_KEY from the .env file

BATCH_SIZE = 90        # stay under the 100/minute free-tier limit
PAUSE_SECONDS = 62     # wait for the per-minute window to reset
PERSIST_DIR = "./chroma_db_rulebook"

# --- Step 1: chunk the rulebook ---
final_chunks = create_chunks()

# --- Step 2: embedding model + vector store on disk ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# --- Step 3: embed and store in batches (respects the rate limit) ---
for start in range(0, len(final_chunks), BATCH_SIZE):
    batch = final_chunks[start:start + BATCH_SIZE]
    while True:
        try:
            vector_db.add_documents(batch)
            break
        except Exception as e:
            if "RESOURCE_EXHAUSTED" not in str(e):
                raise
            print("Rate limit hit, waiting 30s and retrying...")
            time.sleep(30)
    print(f"Indexed {min(start + BATCH_SIZE, len(final_chunks))}/{len(final_chunks)}")
    if start + BATCH_SIZE < len(final_chunks):
        time.sleep(PAUSE_SECONDS)

print(f"Done: {len(final_chunks)} chunks in '{PERSIST_DIR}'")
