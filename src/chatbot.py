"""The chatbot: rewrite the question, retrieve chunks, build the prompt, call the LLM.

The Chatbot object keeps the conversation history, which is what makes it multi-turn.
"""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from config import MAX_HISTORY_TURNS
from llm import get_llm, text_of
from prompts import ANSWER_SYSTEM, REWRITE_SYSTEM, REWRITE_USER
from vectorstore import retrieve


@dataclass
class Response:
    answer: str
    standalone_question: str  # the question actually used for retrieval
    sources: list[Document]  # the chunks given to the LLM


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, start=1))


class Chatbot:
    def __init__(self):
        self.history: list[BaseMessage] = []

    def reset(self) -> None:
        self.history = []

    def _recent_history(self) -> list[BaseMessage]:
        return self.history[-2 * MAX_HISTORY_TURNS :]  # one turn = question + answer

    def _rewrite(self, question: str) -> str:
        """Make a follow-up question understandable without the conversation."""
        if not self.history:
            return question
        transcript = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in self._recent_history()
        )
        messages = [
            SystemMessage(REWRITE_SYSTEM),
            HumanMessage(REWRITE_USER.format(transcript=transcript, question=question)),
        ]
        return text_of(get_llm().invoke(messages)).strip()

    def ask(self, question: str) -> Response:
        # 1. Rewrite follow-ups into standalone questions (multi-turn retrieval)
        standalone = self._rewrite(question)

        # 2. Retrieve the most relevant chunks
        docs = retrieve(standalone)

        # 3. Build the prompt: instructions + context, then history, then the question
        messages = [
            SystemMessage(ANSWER_SYSTEM.format(context=format_context(docs))),
            *self._recent_history(),
            HumanMessage(question),
        ]

        # 4. Generate the answer and remember the turn
        answer = text_of(get_llm().invoke(messages))
        self.history += [HumanMessage(question), AIMessage(answer)]

        return Response(answer=answer, standalone_question=standalone, sources=docs)