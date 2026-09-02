"""Step 3 — Reasoner.

Given the extracted values and the question, the LLM produces a Python
arithmetic expression.  It does NOT compute the result — that is the
executor's job.  This separation eliminates hallucinated arithmetic.
"""

from financial_qa.client.base import LLMClient
from financial_qa.models.pipeline import ExtractedValue
from financial_qa.pipeline.base import SYSTEM_PREAMBLE, build_messages

_SYSTEM = (
    f"{SYSTEM_PREAMBLE}\n\n"
    "Your task: given a question and a list of extracted numerical values, "
    "produce a single Python arithmetic expression (using only +, -, *, /, "
    "parentheses, and numeric literals) that computes the answer.\n\n"
    "Rules:\n"
    "  - Use only the provided values — substitute their 'normalised' floats\n"
    "  - Return ONLY the expression — no variable names, no code, no explanation\n"
    "  - Do not evaluate it yourself"
)


class Reasoner:
    """Produces a Python arithmetic expression from extracted values.

    Args:
        client: LLM client used for reasoning.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def run(self, question: str, values: list[ExtractedValue]) -> str:
        """Generate an arithmetic expression that answers the question.

        Args:
            question: The (rewritten) question to answer.
            values: Numerical values extracted from the context.

        Returns:
            A Python arithmetic expression string.

        Raises:
            ValueError: If the LLM returns an empty expression.
        """
        values_text = "\n".join(
            f"  {v.name}: {v.normalised} (raw: {v.raw})"
            for v in values
        ) or "  (no values provided)"

        user_prompt = (
            f"Question: {question}\n\n"
            f"Available values:\n{values_text}\n\n"
            "Arithmetic expression:"
        )

        response = self._client.complete(build_messages(_SYSTEM, user_prompt))
        expression = response.content.strip()

        if not expression:
            raise ValueError("reasoner returned an empty expression")

        return expression
