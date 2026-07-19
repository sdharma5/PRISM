.PHONY: install format lint typecheck test integration-test smoke validate-registry docs precommit ci clean

PY ?= uv run
# Fall back to the local venv when uv is unavailable.
ifeq ($(shell command -v uv 2>/dev/null),)
PY := .venv/bin/python -m
RUN := .venv/bin/python
else
RUN := uv run python
endif

install:
	uv sync --all-extras || (python3 -m venv .venv && .venv/bin/pip install -e ".[dev]")

format:
	$(PY) ruff format .

lint:
	$(PY) ruff check .

typecheck:
	$(PY) mypy schemas ingestion models training evaluation

test:
	$(PY) pytest tests/unit tests/contract

integration-test:
	$(PY) pytest tests/integration

smoke:
	$(PY) pytest tests/smoke

validate-registry:
	$(RUN) scripts/validate_registry.py

docs:
	$(PY) mkdocs build --strict

precommit:
	$(PY) pre_commit run --all-files

ci: format lint typecheck test validate-registry docs

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache site
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
