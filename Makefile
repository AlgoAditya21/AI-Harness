PYTHON ?= python3

.PHONY: setup run test demo lint

setup:
	@"$(PYTHON)" -m venv .venv
	@.venv/bin/python -m pip install --upgrade pip
	@.venv/bin/python -m pip install -c requirements-dev.lock -e '.[dev]'

run:
	@.venv/bin/python -m harness run $(ARGS)

test:
	@.venv/bin/python -m pytest

demo:
	@.venv/bin/python -m harness demo --allow-host-execution --output .harness-runs

lint:
	@.venv/bin/python -m ruff check harness tests
	@.venv/bin/python -m ruff format --check harness tests
	@.venv/bin/python -m mypy harness
