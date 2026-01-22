.PHONY: help install install-dev format format-check lint test test-cov clean build publish security

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	uv sync --no-dev

install-dev:  ## Install development dependencies
	uv sync

format:  ## Format code with ruff
	uv run ruff format src tests

format-check:  ## Check code formatting without making changes
	uv run ruff format --check src tests

lint:  ## Run linting checks
	uv run ruff check src tests
	uv run ty check src

test:  ## Run tests
	uv run pytest

test-cov:  ## Run tests with coverage
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml

security:  ## Run security checks
	uv add --dev safety pip-audit bandit[toml]
	uv run safety check
	uv run pip-audit
	uv run bandit -r src/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	uv build

publish:  ## Publish to PyPI (requires UV_PUBLISH_TOKEN or TWINE credentials)
	uv publish

check: lint test  ## Run all checks (lint + test)