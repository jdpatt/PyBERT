# Contributing

## Fork/Pull Requests

1. Fork the repository to your own Github account
2. Clone the project to your machine
3. Create a branch locally with a succinct but descriptive name
4. Commit changes to the branch
5. Following any formatting and testing guidelines specific to this repo
6. Push changes to your fork
7. Open a PR in our repository and follow the PR template so that we can efficiently review the changes.

## Tooling

The `Makefile` has most of these commands or features for easy use across developers.  Install the
optional developer tooling run `pip install -r requirements_dev.txt` which conda will install them
via pip into the current environment.  The most useful one is simply `make all` which will test,
format and lint.

### Conda

The traits infrastructure requires conda to build and management its dependencies.  Installing via
pip has not been reliable or workable. See the [Github Wiki](https://github.com/capn-freako/PyBERT/wiki/instant_gratification) for instructions on installing pybert for a normal user or as a developer. All
of the build recipes for conda are under `conda.recipe`.

### Linting

Currently, pybert loosely uses pylint and mypy to catch glaring issues. You can run both with `make lint`.

### Formatting

Formatting in controlled in the `pyproject.toml` file and running `make format` will run autoflake,
isort and black against the codebase.

### Testing

Pytest is used for the test runner and documentation builder. Both pybert's and pyibisami's test suite
will be run with `make tests`.

## Console Entry Points

The easiest way to install and test the `pybert` command is to run `pip install -e .` which will update the script link without having to rebuild and reinstall the conda package.  This
works only because the `install_requires` field is commented out otherwise pip would try and
fail into install the traits infrastructure.  Alternatively, the normal way of calling the package still works: `python -m pybert`.
