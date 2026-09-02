.PHONY: install test lint format run

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ api/ tests/
	mypy src/ api/

format:
	ruff format src/ api/ tests/

run:
	uvicorn api.main:app --reload --port 8000
