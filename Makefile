PYTHON ?= python

.PHONY: lint typecheck test check

lint:
	ruff check .

typecheck:
	mypy

test:
	pytest

check: lint typecheck test
	@echo "All checks passed (ruff -> mypy -> pytest)"
