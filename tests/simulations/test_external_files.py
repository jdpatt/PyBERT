import os
from pathlib import Path

import pytest

from pybert.pybert import PyBERT


def get_ibis_files():
    """Get list of IBIS files from environment variable if it exists.

    If the environment variable PYIBISAMI_TEST_DIR is not set, the test
    will be skipped. We do not want to check in IBIS files to the
    repository, as they are large and may be under license restrictions.
    """
    ibis_dir = os.environ.get("PYIBISAMI_TEST_DIR")
    print(f"IBIS directory: {ibis_dir}")
    if not ibis_dir:
        return []

    ibis_dir = Path(ibis_dir)
    if not ibis_dir.exists():
        return []

    return [str(filepath.absolute()) for filepath in ibis_dir.glob("**/*.ibs")]


@pytest.mark.skip(reason="Skipping external file tests for now")
@pytest.mark.skipif(not get_ibis_files(), reason="Either PYIBISAMI_TEST_DIR is not set or no IBIS files were found")
@pytest.mark.parametrize("ibis_file", get_ibis_files())
def test_external_files(qapp, caplog, ibis_file):
    """Test PyBERT simulation with all external IBIS files in the PYIBISAMI_TEST_DIR environment variable."""

    dut = PyBERT(run_simulation=False)
    is_tx = True if "tx" in ibis_file else False
    if is_tx:
        dut.tx.load_ibis_file(ibis_file)
    else:
        dut.rx.load_ibis_file(ibis_file)
    dut.simulate(block=True)

    assert dut.last_results is not None
    assert "Error" not in caplog.text
