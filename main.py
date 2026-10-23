"""Entry point: python main.py <command>"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Warhammer 40k RAG")
    parser.add_argument(
        "command",
        choices=[
            "parse",
            "chunk",
            "index",
            "inspect",
            "chat",
            "sql",
            "eval-generate",
            "eval",
            "ablation-generate",
            "ablation",
        ],
        help="parse: PDF to Markdown | chunk: show chunk stats | index: build the vector store | "
        "inspect: check retrieval | chat: talk to the bot | sql: test SQL queries on the database | "
        "eval-generate: build the test questions | eval: run the evaluation | "
        "ablation-generate: build the retrieval comparison questions | ablation: compare retrieval settings",
    )
    args = parser.parse_args()

    # imports are inside each branch so 'chat' doesn't load the heavy PDF libraries
    if args.command == "parse":
        from parser import main as run
    elif args.command == "chunk":
        from chunker import main as run
    elif args.command == "index":
        from indexer import main as run
    elif args.command == "inspect":
        from inspect_retrieval import main as run
    elif args.command == "chat":
        from chat import main as run
    elif args.command == "sql":
        from sql_retriever import main as run
    elif args.command == "eval-generate":
        from eval_dataset import main as run
    elif args.command == "eval":
        from evaluation import main as run
    elif args.command == "ablation-generate":
        from retrieval_ablation import generate_questions as run
    elif args.command == "ablation":
        from retrieval_ablation import main as run
    run()


if __name__ == "__main__":
    main()