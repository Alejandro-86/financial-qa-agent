"""Unit tests for pipeline steps — rewriter, extractor, reasoner.

All tests use the StubClient to avoid real LLM calls.
"""

import pytest
from financial_qa.client.base import LLMClient, LLMMessage, LLMResponse, Role
from financial_qa.models.conversation import ConversationTurn
from financial_qa.pipeline.rewriter import Rewriter
from financial_qa.pipeline.extractor import Extractor
from financial_qa.pipeline.reasoner import Reasoner


class StubClient(LLMClient):
    """Returns a predetermined sequence of responses."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)

    def complete(self, messages: list[LLMMessage], **kwargs: object) -> LLMResponse:
        return LLMResponse(
            content=next(self._replies),
            model="stub",
            input_tokens=10,
            output_tokens=5,
        )


# ─── Rewriter ────────────────────────────────────────────────────────────────

class TestRewriter:
    def test_rewrites_question_with_context(self) -> None:
        client = StubClient(["What was the revenue growth rate?"])
        rewriter = Rewriter(client)
        history = [ConversationTurn(question="What was revenue?", answer="$14.5M")]
        result = rewriter.run("How did it change?", history=history)
        assert "revenue" in result.lower()

    def test_first_turn_returns_question_unchanged(self) -> None:
        client = StubClient(["What was revenue in Q2?"])
        rewriter = Rewriter(client)
        result = rewriter.run("What was revenue in Q2?", history=[])
        assert result == "What was revenue in Q2?"

    def test_rewriter_passes_history_to_llm(self) -> None:
        replies = ["What was the Q2 revenue growth?"]
        client = StubClient(replies)
        rewriter = Rewriter(client)
        history = [
            ConversationTurn(question="Q1 revenue?", answer="10M"),
            ConversationTurn(question="Q2 revenue?", answer="14.5M"),
        ]
        result = rewriter.run("How did it change?", history=history)
        assert isinstance(result, str)
        assert len(result) > 0


# ─── Extractor ───────────────────────────────────────────────────────────────

class TestExtractor:
    def test_extracts_values_from_llm_response(self) -> None:
        stub_json = '{"values": [{"name": "revenue_q2", "raw": "$14.5M", "normalised": 14.5}]}'
        client = StubClient([stub_json])
        extractor = Extractor(client)
        values = extractor.run("What is revenue?", context="Revenue was $14.5M")
        assert len(values) == 1
        assert values[0].name == "revenue_q2"
        assert values[0].normalised == pytest.approx(14.5)

    def test_returns_empty_list_when_no_values(self) -> None:
        client = StubClient(['{"values": []}'])
        extractor = Extractor(client)
        values = extractor.run("What is the CEO's name?", context="CEO is John")
        assert values == []

    def test_handles_malformed_json_gracefully(self) -> None:
        client = StubClient(["not valid json"])
        extractor = Extractor(client)
        values = extractor.run("question", context="context")
        assert values == []


# ─── Reasoner ────────────────────────────────────────────────────────────────

class TestReasoner:
    def test_returns_arithmetic_expression(self) -> None:
        client = StubClient(["(14.5 - 12.0) / 12.0 * 100"])
        reasoner = Reasoner(client)
        from financial_qa.models.pipeline import ExtractedValue
        values = [
            ExtractedValue(name="revenue_q2", raw="$14.5M", normalised=14.5),
            ExtractedValue(name="revenue_q1", raw="$12.0M", normalised=12.0),
        ]
        expr = reasoner.run("What is the growth rate?", values=values)
        assert expr == "(14.5 - 12.0) / 12.0 * 100"

    def test_strips_whitespace_from_expression(self) -> None:
        client = StubClient(["  14.5 / 12.0  "])
        reasoner = Reasoner(client)
        expr = reasoner.run("ratio?", values=[])
        assert expr == "14.5 / 12.0"

    def test_raises_if_expression_is_empty(self) -> None:
        client = StubClient([""])
        reasoner = Reasoner(client)
        with pytest.raises(ValueError, match="empty"):
            reasoner.run("question?", values=[])
