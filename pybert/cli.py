"""Command Line Interface for pybert."""
import sys

import click

from pybert.logger import setup_logging
from pybert.pybert import PyBERT
from pybert.view import traits_view


@click.group(invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
@click.version_option()
@click.option("--config", "-c", type=click.Path(exists=True), help="Settings to run pybert with.")
@click.option("--results", "-r", type=click.Path(exists=True), help="Recall previous results when opening pybert.")
def cli(ctx, config, results):
    """Command line interface for pybert."""
    setup_logging()
    if ctx.invoked_subcommand is None:
        app = PyBERT(run_simulation=False, gui=True)

        if config:
            app.load_configuration(config)

        # Either load old results or run the simulation to fill the plots.
        if results:
            app.load_results(results)
        else:
            app.run_simulation(inital_run=True)

        # Create the GUI.
        app.configure_traits(view=traits_view)


@cli.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("config", type=click.Path(exists=True))
def simulate(config):
    """Load the configuration, simulate it and save the results.

    The results will be saved into a file with the same location as the config but with the
    .pybert_data extension.
    """

    app = PyBERT(run_simulation=False, gui=False)
    app.load_configuration(config)
    app.run_simulation(inital_run=True)
    app.save_results(config.with_suffix(".pybert_data"))


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
