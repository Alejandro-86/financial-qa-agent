"""Step 1 — Question Rewriter.

Reformulates the current question so it is self-contained, incorporating
relevant context from prior conversation turns.  On turn 1 (no history)
the question is returned as-is.
"""

from financial_qa.client.base import LLMClient
from financial_qa.models.conversation import ConversationTurn
from financial_qa.pipeline.base import SYSTEM_PREAMBLE, build_messages

_SYSTEM = (
    f"{SYSTEM_PREAMBLE}\n\n"
    "Your task: rewrite the user's question so it is fully self-contained "
    "given the conversation history. Do not answer the question — only rewrite it. "
    "Return the rewritten question as a single sentence."
)


class Rewriter:
    """Rewrites a question to be self-contained given conversation history.

    Args:
        client: LLM client used to generate the rewrite.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def run(self, question: str, history: list[ConversationTurn]) -> str:
        """Rewrite the question with prior-turn context.

        Args:
            question: The raw question for the current turn.
            history: Completed turns preceding this one.

        Returns:
            A self-contained rewritten question string.
        """
        if not history:
            return question

        history_text = "\n".join(
            f"Q{i + 1}: {t.question}\nA{i + 1}: {t.answer}"
            for i, t in enumerate(history)
        )
        user_prompt = (
            f"Conversation so far:\n{history_text}\n\n"
            f"Current question: {question}\n\n"
            "Rewrite the current question to be fully self-contained:"
        )

        response = self._client.complete(build_messages(_SYSTEM, user_prompt))
        return response.content.strip()
