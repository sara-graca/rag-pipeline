"""Generate the evaluation dataset (a small set of questions with reference answers).

Rulebook questions: random chunks of the rulebook are shown to the LLM, which writes a
question that chunk can answer, plus a short reference answer.
Database questions: same idea, but with a few random rows of a random table.

The result is saved to eval/questions.json. Read it before you evaluate: the LLM
sometimes writes a weak question, and you can edit or delete those by hand.
"""

import json
import random
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from chunker import create_chunks
from config import EVAL_N_RULEBOOK, EVAL_N_SQL, EVAL_QUESTIONS_PATH, EVAL_SEED
from llm import get_llm, text_of
from router import DATABASE, RULEBOOK
from sql_retriever import connect, fix_text

MIN_CHUNK_CHARS = 300  # shorter chunks (headings, fragments) make poor questions
SAMPLE_ROWS = 3  # rows of a table shown to the LLM
MAX_CELL_CHARS = 300  # long text values are cut to this length
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 30

RULEBOOK_QA_SYSTEM = """You write test questions for a chatbot that answers from the Warhammer 40,000 Core Rules.
You are given one excerpt of the rulebook. Write ONE question that can be answered
using only that excerpt, and a short reference answer (1-3 sentences) based on it.

- The question must make sense on its own. Never mention "the excerpt" or "the text".
- Do not ask about a specific faction or unit.
Return only JSON, like: {"question": "...", "answer": "..."}"""

SQL_QA_SYSTEM = """You write test questions for a chatbot that answers from a database of Warhammer 40,000 factions and units.
You are given a table's columns and a few of its rows. Write ONE question that can be
answered from these rows, and a short reference answer based on them.

- Name the specific unit, weapon, faction or item the question is about, the way a player would.
- The question must make sense on its own. Never mention "the table" or "the rows".
Return only JSON, like: {"question": "...", "answer": "..."}"""


def call_with_retry(fn, *args, **kwargs):
    """Call fn, waiting and retrying when the API says the rate limit was hit."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" not in str(e) or attempt == MAX_RETRIES - 1:
                raise
            print(f"Rate limit hit, waiting {RETRY_WAIT_SECONDS}s and retrying...")
            time.sleep(RETRY_WAIT_SECONDS)


def ask_llm(system: str, user: str) -> str:
    messages = [SystemMessage(system), HumanMessage(user)]
    return text_of(call_with_retry(get_llm().invoke, messages))


def parse_json(text: str) -> dict:
    """Pull the JSON object out of an LLM reply (it may add code fences or extra words)."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text[:100]!r}")
    return json.loads(match.group(0))


def generate_rulebook_items(n: int, rng: random.Random) -> list[dict]:
    chunks = [c for c in create_chunks() if len(c.page_content) >= MIN_CHUNK_CHARS]
    items = []
    for chunk in rng.sample(chunks, min(n, len(chunks))):
        try:
            qa = parse_json(ask_llm(RULEBOOK_QA_SYSTEM, chunk.page_content))
            items.append(
                {
                    "question": qa["question"].strip(),
                    "expected_route": RULEBOOK,
                    "reference_answer": qa["answer"].strip(),
                    "source_text": chunk.page_content,  # used to check retrieval found it
                }
            )
        except (ValueError, KeyError) as e:
            print(f"Skipped a rulebook chunk: {e}")
    return items


def sample_table_rows(table: str, rng: random.Random) -> tuple[list[str], list[tuple]]:
    conn = connect()
    try:
        cursor = conn.execute(f'SELECT * FROM "{table}" LIMIT 500')
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
    finally:
        conn.close()
    return columns, rng.sample(rows, min(SAMPLE_ROWS, len(rows)))


def generate_sql_items(n: int, rng: random.Random) -> list[dict]:
    conn = connect()
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    finally:
        conn.close()

    items = []
    order = []
    while len(order) < n and tables:  # spread the questions over the tables
        shuffled = tables[:]
        rng.shuffle(shuffled)
        order += shuffled
    for table in order[:n]:
        columns, rows = sample_table_rows(table, rng)
        if not rows:
            continue
        lines = [
            " | ".join(str(fix_text(v))[:MAX_CELL_CHARS].replace("\n", " ") for v in row) for row in rows
        ]
        prompt = f"Table: {table}\nColumns: {', '.join(columns)}\nRows:\n" + "\n".join(lines)
        try:
            qa = parse_json(ask_llm(SQL_QA_SYSTEM, prompt))
            items.append(
                {
                    "question": qa["question"].strip(),
                    "expected_route": DATABASE,
                    "reference_answer": qa["answer"].strip(),
                    "source_table": table,
                }
            )
        except (ValueError, KeyError) as e:
            print(f"Skipped table {table}: {e}")
    return items


def load_questions(path=EVAL_QUESTIONS_PATH) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run 'python main.py eval-generate' first.")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    rng = random.Random(EVAL_SEED)  # fixed seed: the same chunks and rows are picked each time
    items = generate_rulebook_items(EVAL_N_RULEBOOK, rng) + generate_sql_items(EVAL_N_SQL, rng)
    for i, item in enumerate(items, start=1):
        item["id"] = f"q{i}"

    EVAL_QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_QUESTIONS_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(items)} questions to {EVAL_QUESTIONS_PATH}")
    for item in items:
        print(f"  {item['id']} [{item['expected_route']}] {item['question']}")


if __name__ == "__main__":
    main()
