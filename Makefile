# Targets are filled in as each phase lands. Run on Linux/macOS/CI (on Windows use WSL or run the commands by hand).
PY ?= python

.PHONY: install lint test data train serve

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests
	mypy src

test:
	pytest --cov --cov-report=term-missing

data:
	@echo "data pipeline arrives in Phase 1"

train:
	@echo "training arrives in Phase 5"

serve:
	@echo "serving arrives in Phase 8"
