"""Chat with the rulebook in the terminal.

Commands: 'quit' to exit, 'reset' to start a new conversation, 'debug' to toggle
showing the rewritten question and the retrieved chunks.
"""

from rich.console import Console
from rich.markdown import Markdown

from chatbot import Chatbot

console = Console()


def main():
    bot = Chatbot()
    debug = False
    console.print("Ask a question about the rules ('quit', 'reset', 'debug').", style="bold")

    while True:
        question = console.input("\n[bold cyan]You:[/] ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "reset":
            bot.reset()
            console.print("Conversation reset.", style="yellow")
            continue
        if question.lower() == "debug":
            debug = not debug
            console.print(f"Debug {'on' if debug else 'off'}.", style="yellow")
            continue

        with console.status("Thinking..."):
            response = bot.ask(question)

        if debug:
            console.print(f"\nsearched for: {response.standalone_question}", style="dim", markup=False)
            for i, doc in enumerate(response.sources, start=1):
                preview = doc.page_content[:150].replace("\n", " ")
                console.print(f"[{i}] {preview}...", style="dim", markup=False)

        console.print("\n[bold green]Bot:[/]")
        console.print(Markdown(response.answer))


if __name__ == "__main__":
    main()