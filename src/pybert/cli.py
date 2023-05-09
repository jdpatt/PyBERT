"""Main Entry Point for the PyBERT."""
from pathlib import Path

import click

from pybert import __version__
from pybert.gui.pybert import TRAITS_VIEW
from pybert.logger import setup_logger
from pybert.pybert import PyBERT


@click.group(invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
@click.version_option(version=__version__)
@click.option("--config-file", "-c", type=click.Path(exists=True), help="Load an existing configuration file.")
@click.option("--results", "-r", type=click.Path(exists=True), help="Load results from a prior run.")
@click.option("--verbose", "-v", default=False, is_flag=True, help="Enable debug prints on the terminal console.")
def cli(ctx, config_file, results, verbose):
    """Serial communication link bit error rate tester."""
    setup_logger("pybert.log", console_debug=verbose)

    if ctx.invoked_subcommand is None:  # No sub-command like `sim` given open the GUI like default.
        pybert = PyBERT()

        # Load any user provided files before opening the GUI.
        if config_file:
            pybert.load_configuration(config_file)
        if results:
            pybert.load_results(results)

        # Show the GUI.
        pybert.configure_traits(view=TRAITS_VIEW)


@cli.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("config-file", type=click.Path(exists=True))
@click.option("--results", "-r", type=click.Path(), help="Override the results filename.")
def sim(config_file, results):
    """Run a simulation without opening the GUI.

    Will load the CONFIG_FILE from the given filepath, run the
    simulation and then save the results into a file with the same name
    but a different extension as the configuration file.
    """
    pybert = PyBERT(run_simulation=False, gui=False)
    pybert.load_configuration(config_file)
    pybert.simulate(initial_run=True, update_plots=True)
    if not results:
        results = Path(config_file).with_suffix(".pybert_data")
    pybert.save_results(results)
