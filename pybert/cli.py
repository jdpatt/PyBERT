"""Command Line Interface for pybert."""
from pathlib import Path

import click

from pybert.control import my_run_simulation
from pybert.logger import setup_logging
from pybert.pybert import PyBERT
from pybert.view import traits_view


@click.group(invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
@click.version_option()
@click.option("--config", "-c", type=click.Path(exists=True), help="Load with a existing configuration.")
@click.option("--results", "-r", type=click.Path(exists=True), help="Load with results from a prior run.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug printing.")
def cli(ctx, config, results, verbose):
    """Command line interface for pybert.

    Provides a simple way to start pybert with non-default settings.
    Sub-commands add additional features such as simulation in a headless fashion.
    """
    if ctx.invoked_subcommand is None:
        setup_logging(verbose)
        pybert = PyBERT()

        if config:
            pybert.load_configuration(config)
        if results:
            pybert.load_results(results)

        # Create the GUI.
        pybert.configure_traits(view=traits_view)


@cli.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("config", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Enable debug printing.")
def simulate(config, verbose):
    """Run a simulation using an existing configuration.

    The results will be saved in a file with the same name but with the results extension.
    """
    setup_logging(verbose)
    pybert = PyBERT(run_simulation=False, gui=False)
    pybert.load_configuration(config)
    my_run_simulation(pybert, initial_run=True, update_plots=False)
    pybert.save_results(Path(config).with_suffix(".pybert_data"))
