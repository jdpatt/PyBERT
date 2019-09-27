.PHONY: tox clean test tests lint etags conda-build conda-skeleton chaco enable pyibis-ami pybert

tox:
	tox --parallel all

lint:
	tox -e lint

tests: test

test:
	tox -e py37

docs:
	# Docs doesn't rely on docker but does require tox to be installed via pip.
	tox -e docs

clean:
	rm -rf .tox docs/_build/ .pytest_cache .venv .mypy_cache
