all: help
lint:
	ruff check
	ruff format --check
format:
	ruff check --fix
	ruff format

######################
# HELP
######################

help:
	@echo '----'
	@echo 'lint                         - run linters'
	@echo 'format                       - run code formatters'
