"""Unit tests for the provider-agnostic LLM client.

Uses a stub implementation to verify the interface contract without
making real API calls.
"""

import pytest

from financial_qa.client.base import LLMClient, LLMMessage, LLMResponse, Role


class StubClient(LLMClient):
    """Stub that returns a fixed response — no API calls."""

    def __init__(self, reply: str = "stub reply") -> None:
        self._reply = reply
        self.calls: list[list[LLMMessage]] = []

    def complete(self, messages: list[LLMMessage], **kwargs: object) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(content=self._reply, model="stub", input_tokens=10, output_tokens=5)


class TestLLMMessage:
    def test_user_message(self) -> None:
        msg = LLMMessage(role=Role.USER, content="hello")
        assert msg.role == Role.USER
        assert msg.content == "hello"

    def test_assistant_message(self) -> None:
        msg = LLMMessage(role=Role.ASSISTANT, content="reply")
        assert msg.role == Role.ASSISTANT

    def test_system_message(self) -> None:
        msg = LLMMessage(role=Role.SYSTEM, content="you are helpful")
        assert msg.role == Role.SYSTEM

    def test_content_cannot_be_empty(self) -> None:
        with pytest.raises(ValueError):
            LLMMessage(role=Role.USER, content="")


class TestLLMResponse:
    def test_response_stores_content_and_tokens(self) -> None:
        r = LLMResponse(content="answer", model="claude", input_tokens=100, output_tokens=20)
        assert r.content == "answer"
        assert r.total_tokens == 120

    def test_total_tokens_is_sum(self) -> None:
        r = LLMResponse(content="x", model="m", input_tokens=50, output_tokens=30)
        assert r.total_tokens == 80


class TestStubClient:
    def test_complete_returns_response(self) -> None:
        client = StubClient("42.5")
        resp = client.complete([LLMMessage(role=Role.USER, content="what is 2+2?")])
        assert resp.content == "42.5"

    def test_complete_records_call(self) -> None:
        client = StubClient()
        messages = [LLMMessage(role=Role.USER, content="hello")]
        client.complete(messages)
        assert len(client.calls) == 1
        assert client.calls[0] == messages

    def test_complete_called_multiple_times(self) -> None:
        client = StubClient()
        for _ in range(3):
            client.complete([LLMMessage(role=Role.USER, content="x")])
        assert len(client.calls) == 3


class TestClientFactory:
    def test_factory_raises_for_unknown_provider(self) -> None:
        from financial_qa.client.factory import make_client
        with pytest.raises(ValueError, match="unknown provider"):
            make_client(provider="unknown", model="x", api_key="k")

    def test_factory_raises_for_missing_api_key(self) -> None:
        from financial_qa.client.factory import make_client
        with pytest.raises(ValueError, match="api_key"):
            make_client(provider="anthropic", model="claude-sonnet-4-6", api_key="")
