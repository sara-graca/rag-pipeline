# Warhammer 40,000 RAG Chatbot

A multi-turn chatbot that answers questions about **Warhammer 40,000** from two sources:

- **The Core Rules PDF**, through retrieval-augmented generation (RAG) over a vector store.
- **A SQLite database of factions and units** (datasheets, weapons, abilities, points, stratagems...), through LLM-written, read-only SQL.

A router decides for each question whether it needs the rulebook, the database, or **both**, so text-only, database-only and combined questions are all supported. Answers are grounded in what was retrieved: if nothing relevant is found, the bot says so instead of guessing.

LLM backend: Google AI Studio (Gemini), through LangChain.

## Authors

- Manuela Pacheco
- Sara Graça

## Quick start

**Requirements:** [Pixi](https://pixi.sh) and a Google AI Studio API key.

We use Pixi instead of uv or poetry because it installs conda-forge and PyPI packages into one locked environment (`pixi.lock`), which makes the heavy dependencies of `marker-pdf` easier to reproduce on Windows and Linux.

```bash
git clone <repo-url>
cd <repo>
pixi install
echo "GOOGLE_API_KEY=your_api_key_here" > .env        # then put your key in .env: GOOGLE_API_KEY=...
```

**Data.** The parsed rulebook (`data/parsed/rulebook_final.md`) and the ready-built vector index (`data/chroma_db_rulebook/`) are committed, so the `parse` and `index` steps can be skipped. The two database files are not in the repo and must be placed here:

```text
data/raw/wahadb.sqlite
data/raw/SQL_AGENT_DATA_DICTIONARY.md
```

**Chat:**

```bash
pixi run python main.py chat
```

In the chat: `debug` shows the rewritten question, the route, and the retrieved chunks or SQL; `reset` starts a new conversation; `quit` exits.

**Rebuilding (optional).** Only needed if you change the parsing or chunking settings:

```bash
pixi run python main.py parse    # PDF -> Markdown (slow, runs marker-pdf; needs data/raw/Core Rules.pdf)
pixi run python main.py index    # re-embeds the chunks; takes several minutes because of free-tier rate limits
```

The committed index was built from the committed Markdown with chunk size 800 and overlap 100 (`src/config.py`). The evaluation questions store the exact text of their source chunk, so if you rebuild with other settings, regenerate them with `eval-generate` and `ablation-generate`.

## How it works

```text
user question
     │
     ▼
1. Rewrite follow-ups into standalone questions (only if there is history)
     │
     ▼
2. Router (LLM, structured output): RULEBOOK | DATABASE | BOTH
     │            │                       │
 RULEBOOK     DATABASE                  BOTH (one sub-question for each source)
     │            │                       │
 vector/hybrid   LLM writes a        both channels run
 search (Chroma) read-only SELECT,
     │           runs it on SQLite
     │           (1 retry on error)
     └────────────┴───────────┬───────────┘
                              ▼
3. LLM answers using only the retrieved context
```

## Technical choices and why

**Parsing.** The rulebook is converted with both `marker-pdf` and `PyMuPDF`, then merged page by page. Marker gives better structure (headers, tables) but sometimes drops text, so for any page where it captured less than 85% of PyMuPDF's words, PyMuPDF's version is used instead.

**Chunking.** The Markdown is split on headers first, then into chunks of 800 characters with 100 of overlap, so a chunk never mixes two sections. The header path (`Section > Subsection`) is added at the start of each chunk, which keeps long sections identifiable and lets the bot cite the section it answers from.

**Retrieval.** Chunks are embedded with Gemini and stored in ChromaDB, and the top 8 are retrieved. We picked `k=8` because it had the best coverage in the ablation described in the next section. Hybrid search (vector plus BM25, merged with reciprocal rank fusion) is implemented and can be switched on with `SEARCH_TYPE`, but it gave no gain on our questions, so vector search is the default.

**Router.** A single LLM call with structured output picks RULEBOOK, DATABASE or BOTH, and also writes a standalone sub-question for each source it needs. If the router returns nothing valid, the question goes to the rulebook.

**Text-to-SQL.** The prompt includes the data dictionary, the live schema with a few sample rows, and the list of factions. The generated query can never change the data: the database is opened read-only, only `SELECT` and `WITH` are accepted, and results are capped at 50 rows. If a query fails, the error is sent back to the LLM for one correction.

**Multi-turn.** The bot keeps the last 5 question/answer pairs. Follow-ups are rewritten into standalone questions before routing and retrieval, because the router and retrievers only see one question at a time.

**Models.** Chunks and questions are embedded with `gemini-embedding-001` (set once in `config.py`, so both always use the same model). Rewriting, routing and answering use `gemini-3.5-flash-lite`, and SQL generation uses `gemini-3.6-flash` because it needs more reasoning. Answers are judged by a different model (`gemini-3.7-flash`), so the bot does not grade itself.

## Evaluation

The evaluation has three steps, each building on the previous one:

1. **Retrieval ablation:** tune the retrieval settings in isolation (cheap, no chat model).
2. **End-to-end evaluation:** run the whole pipeline with the chosen settings (`TOP_K = 8`).
3. **RAGAS:** score the saved end-to-end results with a second, independent set of metrics.

### 1. Retrieval ablation

```bash
pixi run python main.py ablation-generate
pixi run python main.py ablation
```

One setting is changed at a time from the baseline (chunk size 800, overlap 100, **k=5**, vector search), on 40 generated rulebook questions, with no chat-model calls. A question counts as a hit if at least 50% of its source chunk's sentences appear in the retrieved text (this works across different chunk sizes, unlike an exact chunk match). Average context size is shown because more context costs more tokens.

| Experiment | Hits (of 40) | Avg. context (chars) |
|---|---:|---:|
| baseline (800 / 100 / k=5 / vector) | 38/40 | 2,691 |
| k=3 | 37/40 | 1,591 |
| k=8 | 40/40 | 4,235 |
| chunk size 400 | 37/40 | 1,619 |
| chunk size 1200 | 39/40 | 3,272 |
| overlap 0 | 38/40 | 2,687 |
| hybrid search | 38/40 | 2,850 |

`k=8` was the best-performing setting (40/40 vs 38/40 for the baseline), at the cost of about 57% more context than `k=5`. So `TOP_K` was changed from 5 to 8 and **all following results use `k=8`**. It is the best of the values we tried (3, 5, 8), not a proven optimum: we did not test larger values, and the gain is 2 questions out of 40. Small chunks (400) lose context, and hybrid search does not help on this question set.

### 2. End-to-end evaluation

```bash
pixi run python main.py eval-generate   # generated questions (rulebook chunks and database rows)
pixi run python main.py eval            # run everything; resumable
```

**Dataset** (`eval/`): 10 LLM-generated questions with reference answers (`questions.json`: 6 from rulebook chunks, 4 from database rows) and 15 hand-written ones (`manual_questions.json`), 25 in total. The manual set covers the four benchmark questions from the course, multi-turn follow-ups, an unanswerable question, and 8 cross-source (BOTH) questions.

**Metrics:**

- **Routing accuracy:** did the router pick the expected source? Reported overall and per route.
- **Retrieval hit rate:** was the exact chunk the question came from among the retrieved ones? Only the 6 generated rulebook questions store their source chunk, so only they are measured. (This is a stricter exact-match check than the ablation's sentence coverage, so the two numbers are not directly comparable.)
- **SQL success rate:** did the generated query run and return rows?
- **Answer correctness:** an LLM judge (a different model from the bot) compares the answer with the reference answer.
- **Multi-turn correctness:** same, for questions with a history.

These checks are mapped to this system's own failure points (router, retriever, SQL, answer). They need about one extra LLM call per question, which fits the free-tier quota (a few dozen requests per day per model). The script is resumable across days for the same reason.

#### Results (`TOP_K = 8`)

| Metric | Overall | RULEBOOK | DATABASE | BOTH |
|---|---|---|---|---|
| Routing accuracy | 24/25 (96%) | 11/11 (100%) | 6/6 (100%) | 7/8 (88%) |
| Retrieval hit rate | 6/6 (100%) | 6/6 (100%) | n/a | n/a |
| SQL success rate | 13/13 (100%) | n/a | 6/6 (100%) | 7/7 (100%) |
| Answer correctness | 23/25 (92%) | 11/11 (100%) | 6/6 (100%) | 6/8 (75%) |
| Multi-turn correctness | 3/4 (75%) | 1/1 (100%) | 1/1 (100%) | 1/2 (50%) |

SQL success is measured on the DATABASE and BOTH questions that were routed correctly (13 questions). Retrieval hit rate is measured on the 6 generated rulebook questions.

**Note on q8.** In the first full run, q8 failed (22/25 answer correctness, 12/13 SQL success). The cause was a bug in our pipeline, which we fixed with a change to the SQL prompt (see the analysis below) and then re-ran q8. The table above shows the final results. Because this fix was made after looking at the evaluation results, the q8 score is not a clean held-out measurement.

#### Failures and analysis

Two of the 25 questions fail answer correctness (b2 and b8). q8 failed in the first run and is included because it explains a real bug.

- **Broken text encoding in the database (q8, DATABASE; fixed).** Asked for the point cost of the Starflare Ignition System for the T'au Empire, the LLM wrote `LIKE '%T’au Empire%'` with a curly apostrophe. The database stores that character double-encoded (`Tâ\x80\x99au Empire`; hex `54C3A2C280C299...`). `fix_text()` repairs this text before showing it to the LLM, so the model saw a clean `T’au Empire`, but the query runs against the raw stored text, where the curly apostrophe does not exist. It returned 0 rows and the bot abstained. The model behaved sensibly: the mismatch was between what we showed it and what the query ran against. We fixed it by adding a rule to the SQL prompt (replace apostrophes with `%` in `LIKE` patterns), and q8 then passed. A cleaner long-term fix would be to repair the encoding in the database itself.
- **Table values dropped by the PDF parser (b2, BOTH).** Asked for the wound roll needed for Strength 5 against a Carnifex, the SQL channel correctly returned the Carnifex's Toughness (9) and the vector search retrieved the Wound Roll table. But in the parsed Markdown the numbers inside the table cells (2+, 3+, 4+, 5+, 6+) were dropped by `marker-pdf` (only `|<sup>**+**</sup>|` remains), while the page still passed the 85% word-count check against PyMuPDF. Instead of inventing the 5+ rule, the bot reported that the values were missing from the excerpts. This is a parsing limitation, not a retrieval one, so a larger `k` cannot fix it. (On b7, Strength 9 vs Toughness 9, the bot succeeded because a prose example in a "Fast Dice Rolling" chunk says equal Strength and Toughness needs a 4+.)
- **Follow-up routed to one source instead of both (b8, BOTH, multi-turn).** After "How does the Grenade stratagem work?", the follow-up "Does a Carnifex have the keyword required to use it?" was rewritten to "Does a Carnifex have the Grenades keyword required to use the Grenade stratagem?". Since the keyword was now explicit, the router sent it to DATABASE instead of BOTH (our only routing miss). The SQL then looked for a Carnifex with a "Grenade" keyword and returned 0 rows. The answer prompt read this as "information not found" instead of "the unit does not have this keyword".

**Limitations:** the question set is small, so these numbers are a sanity check and not a benchmark. Gemini is run at its default temperature, so scores vary slightly between runs. The judge is an LLM and can be wrong.

### 3. RAGAS (second opinion)

Our main evaluation uses custom metrics because they map to this system’s failure points. RAGAS has no equivalent for the router or the SQL channel, and it only scores questions that have rulebook context, so it cannot evaluate the DATABASE route at all. It also makes several LLM calls per question, which is costly on the free-tier quota. We therefore use RAGAS as a second opinion/complement: an independent, standard set of metrics (faithfulness, context recall, factual correctness) on the rulebook answers, to cross-check our own LLM judge.

```bash
pixi run -e eval python main.py ragas
```

RAGAS reads the saved `eval/results.json` (the bot is not called again) and scores the answers and retrieved contexts with three metrics. It makes several LLM calls per question, so we ran it once, on the 18 questions that retrieve rulebook chunks (11 RULEBOOK and 7 BOTH), rather than on every evaluation run. It runs in the separate `eval` pixi environment and is resumable, like the main evaluation. DATABASE questions are excluded because they have no rulebook context, and b8 was routed to DATABASE, so it has no scores.

| RAGAS metric | Overall (n=18) | RULEBOOK (n=11) | BOTH (n=7) |
|---|---|---|---|
| Faithfulness | 0.84 | 0.87 | 0.80 |
| Context recall | 0.97 | 1.00 | 0.93 |
| Factual correctness | 0.75 | 0.78 | 0.70 |

Where RAGAS and our judge disagree:

- **m3 (unanswerable question).** The bot correctly said it could not find the answer, and our judge marks it correct. RAGAS gives it faithfulness 0.0, because the refusal sentence is not literally in the retrieved chunks. Excluding m3, RULEBOOK faithfulness is 0.96 and overall faithfulness is 0.89.
- **Factual correctness is stricter than our judge.** It also penalises extra claims, and our answers often add relevant detail beyond the reference, so RULEBOOK scores 0.78 even though our judge marks 11/11 correct. Our judge only checks that the key facts are present and not contradicted. We read the RAGAS number as a relative signal, not as an accuracy score.
- **Context recall is high (0.97)**, which agrees with the retrieval results: the retrieved chunks contain what is needed. The weakest answers (b2) came from the parsed text, not from retrieval.

## Configuration

Everything tunable is in `src/config.py`: model names, chunk size and overlap, `TOP_K`, `SEARCH_TYPE` (`"vector"` or `"hybrid"`), history length, SQL limits, and evaluation and RAGAS settings.

## Project structure

```text
main.py                  CLI entry point (python main.py <command>)
src/
  config.py              Paths, model names, settings
  prompts.py             All prompt templates
  parser.py              PDF -> Markdown (marker + PyMuPDF merged)
  chunker.py             Markdown cleaning and header-aware chunking
  indexer.py             Embeds chunks into ChromaDB (rate-limit aware)
  vectorstore.py         Vector store, BM25 index, hybrid search
  router.py              RULEBOOK / DATABASE / BOTH routing
  sql_retriever.py       Text-to-SQL channel
  chatbot.py             Full pipeline and conversation history
  chat.py                Terminal chat UI
  llm.py                 Gemini models and retry helper
  eval_dataset.py        Generates evaluation questions
  evaluation.py          Runs the evaluation
  ragas_eval.py          RAGAS scores on the saved evaluation results
  retrieval_ablation.py  Compares chunking and retrieval settings
  inspect_retrieval.py   Prints retrieved chunks for a few questions
data/parsed/             Parsed rulebook Markdown (committed)
data/chroma_db_rulebook/ Ready-built vector index (committed)
data/raw/                PDF, SQLite database and data dictionary (not committed)
eval/                    Question sets and results
tests/                   pytest suite (LLM calls are faked)
```

## Development

```bash
pixi run test
pixi run pre-commit install    # once after cloning: git hooks are not shared through git
```

Code style: Ruff (line length 120) through pre-commit.

## Known limitations

- The free-tier quota limits how fast indexing and evaluation can run. Quotas also appeared to be shared between neighbouring model versions (e.g. 3.5 with 3.6), so switching models does not give a fresh daily allowance. This is one reason the evaluation is resumable.
- Text-to-SQL can still write a wrong query for unusual questions. The bot then answers from a wrong or empty result, so check the SQL with `debug` on. An empty result can also mean "this unit does not have that property" (see b8), which the bot may report as "not found".
- The database contains some double-encoded text. `fix_text()` hides it from the LLM and the prompt avoids apostrophes in `LIKE` patterns, but other special characters in names could still cause mismatches.
- Tables in the rulebook can lose cell values in parsing (see b2), and images and diagrams are not indexed (only text).