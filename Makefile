.PHONY: all lint format tests clean dev-tools etags conda-build conda-skeleton chaco enable pyibis-ami pybert

all: format tests lint

lint:
	pylint pybert/ tests/; mypy -p pybert --ignore-missing-imports

format:
	autoflake --in-place --remove-all-unused-imports --expand-star-imports \
	--ignore-init-module-imports --recursive pybert/ tests/; isort pybert/ tests/; black pybert/ tests/

tests:
	pytest -vv -n 4 --disable-pytest-warnings tests/
	pytest -vv -n 4 --disable-pytest-warnings PyAMI/tests

clean:
	rm -rf .pytest_cache .tox htmlcov *.egg-info .coverage

# Conda Packaging Commands ----------------------------------------------------------

conda-build: tests
	conda build conda.recipe/pybert

conda-skeleton:
	rm -rf conda.recipe/pybert/ conda.recipe/pyibis-ami/ \
	conda skeleton pypi --noarch-python --output-dir=conda.recipe pybert pyibis-ami

etags:
	etags -o TAGS pybert/*.py

chaco:
	conda build --numpy=1.16 conda.recipe/chaco
	conda install --use-local chaco

enable:
	conda build --numpy=1.16 conda.recipe/enable
	conda install --use-local enable

pyibis-ami: tests
	conda build --numpy=1.16 conda.recipe/pyibis-ami

pyibis-ami_dev:
	conda install -n pybert64 --use-local --only-deps PyAMI/
	conda develop -n pybert64 PyAMI/

pybert: pybert_bld pybert_inst

pybert_bld: tests
	conda build --numpy=1.16 conda.recipe/pybert

pybert_inst:
	conda install --use-local pybert

pybert_dev: pybert_bld
	conda install -n pybert64 --use-local --only-deps pybert
	conda develop -n pybert64 .

# End Conda Packaging Commands ----------------------------------------------------------

dev-tools:
	conda env update --file environment.yml 
