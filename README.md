# financial-qa-agent

A multi-turn financial question-answering system built on a four-step LLM reasoning
pipeline. Surpasses the fine-tuned FinQANet baseline (69.9% vs 68.9%) with zero
model training — prompting and orchestration only.

## Architecture

```
User question
      │
      ▼
┌─────────────┐
│  Rewriter   │  Reformulates the question with prior-turn context
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Extractor  │  Identifies relevant numerical values from the table/text
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Reasoner   │  Produces a Python arithmetic expression (never evaluates it)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Executor   │  AST-validated sandbox evaluates the expression safely
└──────┬──────┘
       │
       ▼
    Answer
```

The model **never performs arithmetic directly** — it emits a Python expression
that the Executor evaluates in a sandboxed AST parser. This eliminates hallucinated
arithmetic as a failure class.

## Evaluation

The evaluation framework segments accuracy by:
- Turn depth (Q1, Q2, Q3+)
- Conversation type (single-op, multi-op, hybrid)
- Operator (add, subtract, multiply, divide, percent-change)

Accuracy improved 46.5% → 69.9% through measurement-driven iteration:
percentage normalisation, cross-turn result injection, few-shot RAG retrieval,
and self-consistency (N=3) sampling.

## Quickstart

```bash
# install
pip install -e ".[dev]"

# copy and fill in API keys
cp .env.example .env

# run tests
make test

# start API
make run
# → http://localhost:8000/docs
```

## Docker

```bash
docker compose up
```

## Project structure

```
src/financial_qa/
├── client/      provider-agnostic LLM client (Anthropic / OpenAI / Groq)
├── models/      Pydantic schemas for all I/O
├── pipeline/    four-step reasoning pipeline
└── eval/        metrics, failure taxonomy, evaluation harness
api/             FastAPI application
tests/
├── unit/        isolated unit tests (no LLM calls)
└── integration/ end-to-end pipeline tests
```

## Provider configuration

Switch provider via environment variable — no code changes:

```bash
LLM_PROVIDER=anthropic LLM_MODEL=claude-sonnet-4-6 make run
LLM_PROVIDER=openai    LLM_MODEL=gpt-4o            make run
LLM_PROVIDER=groq      LLM_MODEL=llama3-70b-8192   make run
```
