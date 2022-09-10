.PHONY: all lint typecheck format tests dev-tools

all: format tests lint

lint: typecheck
	pylint src/pybert/ tests/

typecheck:
	mypy -p pybert --ignore-missing-imports

format:
	autoflake --in-place --remove-all-unused-imports --expand-star-imports \
	--ignore-init-module-imports --recursive src/pybert/ tests/; isort src/pybert/ tests/; black src/pybert/ tests/

tests:
	pytest -vv -n 4 --disable-pytest-warnings tests/

dev-tools:
	pip install -r requirements-dev.txt
	pre-commit install
