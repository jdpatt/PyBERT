"""Main entry into the PyBERT GUI."""
from pybert.logger import setup_logging
from pybert.pybert import PyBERT
from pybert.views import traits_view


def main():
    """Setup the logging infrastructure and kick off the GUI."""
    setup_logging()
    app = PyBERT()
    app.configure_traits(view=traits_view)


if __name__ == "__main__":
    main()
