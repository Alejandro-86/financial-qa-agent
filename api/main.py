"""FastAPI application for the financial QA pipeline."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import AskRequest, AskResponse, HealthResponse
from financial_qa.cache import PredictionCache
from financial_qa.client.factory import make_client
from financial_qa.config import settings
from financial_qa.pipeline.runner import PipelineRunner

_runner: PipelineRunner | None = None
_cache: PredictionCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise the pipeline runner and cache on startup."""
    global _runner, _cache

    api_key = (
        settings.anthropic_api_key
        if settings.llm_provider == "anthropic"
        else settings.openai_api_key
        if settings.llm_provider == "openai"
        else settings.groq_api_key
    )

    client = make_client(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    _runner = PipelineRunner(client)
    _cache = PredictionCache(settings.cache_path)
    yield


app = FastAPI(
    title="financial-qa-agent",
    description="Multi-turn financial QA via a four-step LLM reasoning pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health and active model configuration."""
    return HealthResponse(provider=settings.llm_provider, model=settings.llm_model)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Answer a financial question using the four-step pipeline.

    Returns a cached result if one exists for this conversation turn.
    """
    if _runner is None or _cache is None:
        raise HTTPException(status_code=503, detail="pipeline not initialised")

    turn = len(request.history) + 1

    if _cache.has(request.conversation_id, turn):
        cached = _cache.get(request.conversation_id, turn)
        if cached is not None:
            return AskResponse.from_result(cached, cached=True)

    result = _runner.run(
        question=request.question,
        context=request.context,
        conversation_id=request.conversation_id,
        turn=turn,
        history=request.history,
    )
    _cache.store(result)
    return AskResponse.from_result(result)
