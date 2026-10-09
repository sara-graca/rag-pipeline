"""Rebuild the vector store from scratch: chunk the rulebook, embed, store."""

import shutil
import time

from chunker import create_chunks
from config import PERSIST_DIR
from vectorstore import get_vector_db

BATCH_SIZE = 90  # stay under the 100/minute free-tier limit
PAUSE_SECONDS = 62  # wait for the per-minute window to reset


def main():
    # --- Step 1: chunk the rulebook ---
    final_chunks = create_chunks()
    ids = [f"chunk-{i}" for i in range(len(final_chunks))]

    # --- Step 2: start from an empty store, so re-running never duplicates chunks ---
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    vector_db = get_vector_db()

    # --- Step 3: embed and store in batches (respects the rate limit) ---
    for start in range(0, len(final_chunks), BATCH_SIZE):
        batch = final_chunks[start : start + BATCH_SIZE]
        batch_ids = ids[start : start + BATCH_SIZE]
        while True:
            try:
                vector_db.add_documents(batch, ids=batch_ids)
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


if __name__ == "__main__":
    main()