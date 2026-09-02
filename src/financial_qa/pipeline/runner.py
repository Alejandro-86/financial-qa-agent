"""Orchestrates the four-step pipeline for a single question turn."""

import time
import logging

from financial_qa.client.base import LLMClient
from financial_qa.models.conversation import ConversationTurn
from financial_qa.models.pipeline import PipelineResult, PipelineStep, StepStatus
from financial_qa.pipeline.executor import ExecutionError, Executor
from financial_qa.pipeline.extractor import Extractor
from financial_qa.pipeline.reasoner import Reasoner
from financial_qa.pipeline.rewriter import Rewriter

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Runs the four-step financial QA pipeline for a single turn.

    Steps: rewriter → extractor → reasoner → executor.

    Args:
        client: LLM client for all LLM calls.
    """

    def __init__(self, client: LLMClient) -> None:
        self._rewriter = Rewriter(client)
        self._extractor = Extractor(client)
        self._reasoner = Reasoner(client)
        self._executor = Executor()

    def run(
        self,
        question: str,
        context: str,
        conversation_id: str,
        turn: int,
        history: list[ConversationTurn] | None = None,
    ) -> PipelineResult:
        """Execute the full four-step pipeline.

        Args:
            question: The raw question for this turn.
            context: Financial table or passage.
            conversation_id: Groups this turn with others in the conversation.
            turn: 1-based turn index.
            history: Prior completed turns for context rewriting.

        Returns:
            PipelineResult with answer, expression, and per-step records.
        """
        history = history or []
        steps: list[PipelineStep] = []

        # Step 1 — Rewrite
        rewritten = self._run_step(
            name="rewriter",
            steps=steps,
            fn=lambda: self._rewriter.run(question, history=history),
        )
        if rewritten is None:
            return PipelineResult(
                conversation_id=conversation_id, turn=turn,
                question=question, steps=steps, error="rewriter_failed",
            )

        # Step 2 — Extract
        values = self._run_step(
            name="extractor",
            steps=steps,
            fn=lambda: self._extractor.run(rewritten, context=context),
        )
        if values is None:
            return PipelineResult(
                conversation_id=conversation_id, turn=turn,
                question=question, steps=steps, error="extraction_failed",
            )

        # Step 3 — Reason
        expression = self._run_step(
            name="reasoner",
            steps=steps,
            fn=lambda: self._reasoner.run(rewritten, values=values),
        )
        if expression is None:
            return PipelineResult(
                conversation_id=conversation_id, turn=turn,
                question=question, steps=steps, error="reasoning_failed",
            )

        # Step 4 — Execute
        answer = self._run_step(
            name="executor",
            steps=steps,
            fn=lambda: self._executor.run(expression),
        )
        if answer is None:
            return PipelineResult(
                conversation_id=conversation_id, turn=turn,
                question=question, expression=expression,
                steps=steps, error="execution_failed",
            )

        return PipelineResult(
            conversation_id=conversation_id, turn=turn,
            question=question, answer=answer,
            expression=expression, steps=steps,
        )

    def _run_step(
        self,
        name: str,
        steps: list[PipelineStep],
        fn: object,
    ) -> object:
        """Run a single step, record it, and return its output or None on error."""
        t0 = time.monotonic()
        try:
            output = fn()  # type: ignore[operator]
            latency = (time.monotonic() - t0) * 1000
            steps.append(PipelineStep(
                name=name, status=StepStatus.SUCCESS,
                input="", output=str(output), latency_ms=latency,
            ))
            return output
        except (ExecutionError, ValueError, Exception) as exc:
            latency = (time.monotonic() - t0) * 1000
            logger.warning("step %s failed: %s", name, exc)
            steps.append(PipelineStep(
                name=name, status=StepStatus.FAILED,
                input="", output=None, error=str(exc), latency_ms=latency,
            ))
            return None
