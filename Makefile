.PHONY: install lint format test precommit train audit security

install:
	poetry install
	poetry run pre-commit install

lint:
	poetry run ruff check src/ tests/
	poetry run mypy src/

format:
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

test:
	poetry run pytest

precommit:
	poetry run pre-commit run --all-files

train:
	poetry run python -m maternal_risk.models.train

predict:
	poetry run python -m maternal_risk.models.predict --input data/new.csv

audit:
	poetry run pip-audit

security:
	poetry run bandit -c pyproject.toml -r src/
	poetry run pip-audit
