"""Run the evaluation: send every dataset question through the chatbot and measure how it did.

Metrics:
- routing accuracy: did the router pick the expected channel (RULEBOOK / DATABASE)?
- retrieval hit rate: for rulebook questions that were routed correctly, was the chunk
  the question came from among the retrieved chunks?
- SQL success rate: for database questions that were routed correctly, did the generated
  query run without error and return rows?
- answer correctness: an LLM judge compares the bot's answer with the reference answer.

Results are saved to eval/results.json.
"""

import json
import time

from chatbot import Chatbot
from config import EVAL_PAUSE_SECONDS, EVAL_RESULTS_PATH
from eval_dataset import ask_llm, call_with_retry, load_questions
from router import DATABASE, RULEBOOK

JUDGE_SYSTEM = """You grade a chatbot's answer against a reference answer.

Reply CORRECT if the chatbot's answer contains the key facts of the reference answer
and does not contradict it. Extra detail is fine.
Reply INCORRECT if key facts are missing, if it contradicts the reference, or if it
says it could not find the information.

Start your reply with exactly one word, CORRECT or INCORRECT, then a short reason."""

JUDGE_USER = """Question: {question}

Reference answer: {reference}

Chatbot's answer: {answer}"""


def judge(question: str, reference: str, answer: str) -> tuple[bool, str]:
    """Ask the LLM whether the answer matches the reference. Returns (correct, reply)."""
    reply = ask_llm(JUDGE_SYSTEM, JUDGE_USER.format(question=question, reference=reference, answer=answer))
    verdict = reply.strip().lstrip("*`\"' ").upper()
    return verdict.startswith("CORRECT"), reply.strip()


def evaluate_item(bot, item: dict, judge_fn=None) -> dict:
    """Run one question through the bot and score it."""
    judge_fn = judge_fn or judge
    result = {
        "id": item["id"],
        "question": item["question"],
        "expected_route": item["expected_route"],
        "reference_answer": item["reference_answer"],
        "route": None,
        "route_ok": False,
        "retrieval_hit": None,  # only for rulebook questions that were routed correctly
        "sql_ok": None,  # only for database questions that were routed correctly
        "sql_query": None,
        "answer": None,
        "correct": None,
        "judge_reply": None,
        "error": None,
    }

    bot.reset()  # every question starts without history
    try:
        response = call_with_retry(bot.ask, item["question"])
    except Exception as e:
        result.update(error=str(e), correct=False)
        return result

    result["route"] = response.route
    result["route_ok"] = response.route == item["expected_route"]
    result["answer"] = response.answer

    if result["route_ok"] and item["expected_route"] == RULEBOOK:
        result["retrieval_hit"] = any(doc.page_content == item.get("source_text") for doc in response.sources)
    if response.sql is not None:
        result["sql_query"] = response.sql.query
    if result["route_ok"] and item["expected_route"] == DATABASE:
        result["sql_ok"] = response.sql is not None and response.sql.error is None and bool(response.sql.rows)

    try:
        result["correct"], result["judge_reply"] = judge_fn(item["question"], item["reference_answer"], response.answer)
    except Exception as e:
        result["judge_reply"] = f"judge failed: {e}"  # correct stays None: not counted

    return result


def rate(values: list) -> dict:
    """Pass rate over the values that are not None."""
    counted = [v for v in values if v is not None]
    passed = sum(1 for v in counted if v)
    return {"passed": passed, "total": len(counted), "rate": passed / len(counted) if counted else None}


def summarize(results: list[dict]) -> dict:
    return {
        "n_questions": len(results),
        "n_errors": sum(1 for r in results if r["error"]),
        "routing_accuracy": rate([r["route_ok"] for r in results]),
        "retrieval_hit_rate": rate([r["retrieval_hit"] for r in results]),
        "sql_success_rate": rate([r["sql_ok"] for r in results]),
        "answer_correctness": rate([r["correct"] for r in results]),
    }


def format_rate(stat: dict) -> str:
    if stat["rate"] is None:
        return "n/a"
    return f"{stat['passed']}/{stat['total']} ({stat['rate']:.0%})"


def mark(value) -> str:
    return "-" if value is None else ("yes" if value else "NO")


def print_item(i: int, total: int, result: dict) -> None:
    print(
        f"[{i}/{total}] {result['id']}  expected {result['expected_route']}, got {result['route']}  "
        f"| retrieval: {mark(result['retrieval_hit'])}  sql: {mark(result['sql_ok'])}  "
        f"correct: {mark(result['correct'])}"
    )
    if result["error"]:
        print(f"        error: {result['error']}")


def print_summary(summary: dict) -> None:
    print("\n=== Evaluation summary ===")
    print(f"Questions:          {summary['n_questions']} ({summary['n_errors']} with errors)")
    print(f"Routing accuracy:   {format_rate(summary['routing_accuracy'])}")
    print(f"Retrieval hit rate: {format_rate(summary['retrieval_hit_rate'])}  (rulebook questions routed correctly)")
    print(f"SQL success rate:   {format_rate(summary['sql_success_rate'])}  (database questions routed correctly)")
    print(f"Answer correctness: {format_rate(summary['answer_correctness'])}  (judged by the LLM)")


def main():
    questions = load_questions()
    bot = Chatbot()

    results = []
    for i, item in enumerate(questions, start=1):
        result = evaluate_item(bot, item)
        results.append(result)
        print_item(i, len(questions), result)
        if i < len(questions):
            time.sleep(EVAL_PAUSE_SECONDS)  # stay under the free-tier rate limit

    summary = summarize(results)
    print_summary(summary)

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nFull results (answers, SQL, judge replies) saved to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
