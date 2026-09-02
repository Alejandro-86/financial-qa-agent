"""Shared prompt-building utilities for pipeline steps."""

from financial_qa.client.base import LLMMessage, Role

SYSTEM_PREAMBLE = (
    "You are a precise financial analysis assistant. "
    "Follow instructions exactly. Respond only with what is asked — no explanation."
)


def build_messages(system: str, user: str) -> list[LLMMessage]:
    """Build a minimal two-message chat for a single pipeline step.

    Args:
        system: The system-level instruction for this step.
        user: The user-level prompt containing the actual task.

    Returns:
        Ordered list of LLMMessage ready for ``LLMClient.complete``.
    """
    return [
        LLMMessage(role=Role.SYSTEM, content=system),
        LLMMessage(role=Role.USER, content=user),
    ]
