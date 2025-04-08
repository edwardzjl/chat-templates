all: help
lint:
	uv run ruff check
	uv run ruff format --check
format:
	uv run ruff check --fix
	uv run ruff format
test:
	uv run python -m unittest

######################
# HELP
######################

help:
	@echo '----'
	@echo 'lint                         - run linters'
	@echo 'format                       - run code formatters'
	@echo 'test                         - run unittests'
