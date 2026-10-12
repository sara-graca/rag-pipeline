"""The router: decides if a question is answered from the rulebook or from the database."""

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm, text_of
from prompts import ROUTER_SYSTEM

RULEBOOK = "RULEBOOK"
DATABASE = "DATABASE"


def route(question: str) -> str:
    """Return RULEBOOK or DATABASE. Anything unclear goes to the rulebook."""
    messages = [SystemMessage(ROUTER_SYSTEM), HumanMessage(question)]
    reply = text_of(get_llm().invoke(messages)).strip().upper()
    return DATABASE if DATABASE in reply else RULEBOOK


def main():
    """Type questions and see which channel the router picks."""
    print("Type a question to see where it would be routed ('quit' to exit).")
    while (question := input("\nQuestion: ").strip()).lower() != "quit":
        if question:
            print(route(question))


if __name__ == "__main__":
    main()