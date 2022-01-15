"""Various menus have information or status strings based of the current configuration or results.
"""
from pybert.pybert import PyBERT
from pybert.views.content import performance_info_table

# pylint: disable=redefined-outer-name,protected-access


def test_sweep_info_string_no_sweep_results():
    """Present an empty table to the user."""
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
    """Present a table with just one configuration/row to the user."""
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


def test_performance_info_table():
    """Present a table to the user show the take it takes to complete each component."""
    assert (
        performance_info_table(0, 0, 0, 0, 0, 0, 0)
        == r"""<H2>Performance by Component</H2>
<TABLE border="1">
  <TR align="center">
    <TH>Component</TH><TH>Performance (Msmpls./min.)</TH>
  </TR>
  <TR align="right">
    <TD align="center">Channel</TD><TD>0</TD>
  </TR>
  <TR align="right">
    <TD align="center">Tx Preemphasis</TD><TD>0</TD>
  </TR>
  <TR align="right">
    <TD align="center">CTLE</TD><TD>0</TD>
  </TR>
  <TR align="right">
    <TD align="center">DFE</TD><TD>0</TD>
  </TR>
  <TR align="right">
    <TD align="center">Jitter Analysis</TD><TD>0</TD>
  </TR>
  <TR align="right">
    <TD align="center">Plotting</TD><TD>0</TD>
  </TR>
  <TR align="right">
    <TD align="center"><strong>TOTAL</strong></TD><TD><strong> 0.000</strong></TD>
  </TR>
</TABLE>
"""
    )
