"""Step 2 — Value Extractor.

Identifies numerical values relevant to the question from the financial
context (table or passage).  Returns a list of ExtractedValue objects.
The LLM must respond with JSON — malformed responses return an empty list.
"""

import json
import logging

from financial_qa.client.base import LLMClient
from financial_qa.models.pipeline import ExtractedValue
from financial_qa.pipeline.base import SYSTEM_PREAMBLE, build_messages

logger = logging.getLogger(__name__)

_SYSTEM = (
    f"{SYSTEM_PREAMBLE}\n\n"
    "Your task: identify every numerical value in the context that is relevant "
    "to answering the question. Return a JSON object with a single key 'values', "
    "containing a list of objects each with:\n"
    "  - name: a short snake_case label\n"
    "  - raw: the value exactly as it appears in the context\n"
    "  - normalised: the value as a plain float (millions → millions, percentages → decimals)\n\n"
    "Return ONLY the JSON object — no markdown, no explanation."
)


class Extractor:
    """Extracts normalised numerical values from financial context.

    Args:
        client: LLM client used for extraction.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def run(self, question: str, context: str) -> list[ExtractedValue]:
        """Extract relevant numerical values from the context.

        Args:
            question: The (rewritten) question to answer.
            context: Financial table or passage containing the numbers.

        Returns:
            List of ExtractedValue objects.  Returns empty list on any
            parse failure rather than raising — pipeline handles gracefully.
        """
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        response = self._client.complete(build_messages(_SYSTEM, user_prompt))

        try:
            data = json.loads(response.content)
            return [ExtractedValue(**v) for v in data.get("values", [])]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("extractor: failed to parse LLM response: %s", exc)
            return []
