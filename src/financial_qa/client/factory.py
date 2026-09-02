"""Factory for constructing the correct LLM client from config."""

from financial_qa.client.base import LLMClient


def make_client(provider: str, model: str, api_key: str, **kwargs: object) -> LLMClient:
    """Construct an LLM client for the given provider.

    Args:
        provider: One of 'anthropic', 'openai', 'groq'.
        model: Model identifier string for the provider.
        api_key: API key for the provider.
        **kwargs: Passed through to the client constructor (temperature, max_tokens).

    Returns:
        A concrete LLMClient instance.

    Raises:
        ValueError: If the provider is not recognised or api_key is empty.

    Example:
        >>> client = make_client("anthropic", "claude-sonnet-4-6", api_key="sk-...")
    """
    if not api_key:
        raise ValueError("api_key must not be empty")

    match provider.lower():
        case "anthropic":
            from financial_qa.client.anthropic import AnthropicClient
            return AnthropicClient(model=model, api_key=api_key, **kwargs)  # type: ignore[arg-type]
        case "openai":
            from financial_qa.client.openai import OpenAIClient
            return OpenAIClient(model=model, api_key=api_key, **kwargs)  # type: ignore[arg-type]
        case "groq":
            from financial_qa.client.groq import GroqClient
            return GroqClient(model=model, api_key=api_key, **kwargs)  # type: ignore[arg-type]
        case _:
            raise ValueError(f"unknown provider '{provider}' — expected anthropic, openai or groq")
