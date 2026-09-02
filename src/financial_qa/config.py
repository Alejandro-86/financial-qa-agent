"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the financial-qa-agent.

    All values can be overridden via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Self-consistency sampling
    self_consistency_n: int = 3

    # Provider API keys (optional — only the active provider is required)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""

    # Resumable cache
    cache_path: str = "cache/predictions.jsonl"


settings = Settings()
