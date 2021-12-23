"""Various menus have information or status strings based of the current configuration or results.
"""
from pybert.pybert import PyBERT

# pylint: disable=redefined-outer-name,protected-access


def test_sweep_info_string_no_sweep_results():
    app = PyBERT(run_simulation=False, gui=False)
    assert (
        app.sweep_info
        == r"""<H2>Sweep Results</H2>
<TABLE border="1">
    <TR align="center">
        <TH>Pretap</TH><TH>Posttap</TH><TH>Mean(bit errors)</TH><TH>StdDev(bit errors)</TH>
    </TR>
</TABLE>
"""
    )


def test_sweep_info_string_after_sweep_simulation():
    app = PyBERT(run_simulation=False, gui=False)
    app.sweep_results = [([0, 1, 2, 3], 456, 789)]
    assert (
        app.sweep_info
        == r"""<H2>Sweep Results</H2>
<TABLE border="1">
    <TR align="center">
        <TH>Pretap</TH><TH>Posttap</TH><TH>Mean(bit errors)</TH><TH>StdDev(bit errors)</TH>
    </TR>
    <TR align="center">
        <TD>0</TD><TD>[1, 2, 3]</TD><TD>456</TD><TD>789</TD>
    </TR>
</TABLE>
"""
    )
