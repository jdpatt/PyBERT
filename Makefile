.PHONY: all lint typecheck format tests dev-tools

all: format tests lint

lint: typecheck
	pylint pybert/ tests/

typecheck:
	mypy -p pybert --ignore-missing-imports

format:
	autoflake --in-place --remove-all-unused-imports --expand-star-imports \
	--ignore-init-module-imports --recursive pybert/ tests/; isort pybert/ tests/; black pybert/ tests/
	pushd PyAMI; make format; popd

tests:
	pytest -vv -n 4 --disable-pytest-warnings tests/

dev-tools:
	pre-commit install
