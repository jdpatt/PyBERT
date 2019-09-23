"""
Any static content that gets rendered in the GUI.

Original author: David Banas <capn.freako@gmail.com>"

Original date:   April 15, 2015 (Copied from pybert.py.)

Copyright (c) 2015 David Banas; all rights reserved World wide.
"""
from pybert.utility import safe_log10


def help_menu():
    """Return the content for the help tab of the GUI."""
    return """<H2>PyBERT User's Guide</H2>\n
  <H3>Note to developers</H3>\n
    This is NOT for you. Instead, open 'pybert/doc/_build/html/index.html' in a browser.\n
  <H3>PyBERT User Help Options</H3>\n
    <UL>\n
      <LI>Hover over any user-settable value in the <em>Config.</em> tab, for help message.</LI>\n
      <LI>Peruse the <em>General Tips</em> & <em>Help by Tab</em> section, below.</LI>\n
      <LI>Visit the PyBERT FAQ at: https://github.com/capn-freako/PyBERT/wiki/pybert_faq.</LI>\n
      <LI>Send e-mail to David Banas at capn.freako@gmail.com.</LI>\n
    </UL>\n
  <H3>General Tips</H3>\n
    <H4>Main Window Status Bar</H4>\n
      The status bar, just above the <em>Run</em> button, gives the following information, from left to right:.<p>\n
      (Note: the individual pieces of information are separated by vertical bar, or 'pipe', characters.)\n
        <UL>\n
          <LI>Current state of, and/or activity engaged in by, the program.</LI>\n
          <LI>Simulator performance, in mega-samples per minute. A 'sample' corresponds to a single value in the signal vector being processed.</LI>\n
          <LI>The observed delay in the channel; can be used as a sanity check, if you know your channel.</LI>\n
          <LI>The number of bit errors detected in the last successful simulation run.</LI>\n
          <LI>The average power dissipated by the transmitter, assuming perfect matching to the channel ,no reflections, and a 50-Ohm system impedance.</LI>\n
          <LI>The jitter breakdown for the last run. (Taken at DFE output.)</LI>\n
        </UL>\n
  <H3>Help by Tab</H3>\n
    <H4>Config.</H4>\n
      This tab allows you to configure the simulation.\n
      Hover over any user configurable element for a help message.\n
"""


def about_menu(authors, copy, date, version):
    """Return the author and version of pybert."""
    return (
            f"PyBERT v{version} - a serial communication link design tool, written in Python.\n\n"
            f"{authors}\n"
            f"{date}   \n"
            f"{copy};  \n"
            "All rights reserved World wide."
        )


def jitter_rejection_menu(jitter):
    """Return the content for the jitter rejection tab of the GUI.  We need to calculate the
    jitter rejection ratios as well."""

    isi_chnl = jitter[""].isi_chnl * 1.0e12
    dcd_chnl = jitter[""].dcd_chnl * 1.0e12
    pj_chnl = jitter[""].pj_chnl * 1.0e12
    rj_chnl = jitter[""].rj_chnl * 1.0e12
    isi_tx = jitter[""].isi_tx * 1.0e12
    dcd_tx = jitter[""].dcd_tx * 1.0e12
    pj_tx = jitter[""].pj_tx * 1.0e12
    rj_tx = jitter[""].rj_tx * 1.0e12
    isi_ctle = jitter[""].isi_ctle * 1.0e12
    dcd_ctle = jitter[""].dcd_ctle * 1.0e12
    pj_ctle = jitter[""].pj_ctle * 1.0e12
    rj_ctle = jitter[""].rj_ctle * 1.0e12
    isi_dfe = jitter[""].isi_dfe * 1.0e12
    dcd_dfe = jitter[""].dcd_dfe * 1.0e12
    pj_dfe = jitter[""].pj_dfe * 1.0e12
    rj_dfe = jitter[""].rj_dfe * 1.0e12

    isi_rej_tx = calc_reject(isi_chnl, isi_tx)
    dcd_rej_tx = calc_reject(dcd_chnl, dcd_tx)
    isi_rej_ctle = calc_reject(isi_tx, isi_ctle)
    dcd_rej_ctle = calc_reject(dcd_tx, dcd_ctle)
    pj_rej_ctle = calc_reject(pj_tx, pj_ctle)
    rj_rej_ctle = calc_reject(rj_tx, rj_ctle)
    isi_rej_dfe = calc_reject(isi_ctle, isi_dfe)
    dcd_rej_dfe = calc_reject(dcd_ctle, dcd_dfe)
    pj_rej_dfe = calc_reject(pj_ctle, pj_dfe)
    rj_rej_dfe = calc_reject(rj_ctle, rj_dfe)
    isi_rej_total = calc_reject(isi_chnl, isi_dfe)
    dcd_rej_total = calc_reject(dcd_chnl, dcd_dfe)
    pj_rej_total = calc_reject(pj_tx, pj_dfe)
    rj_rej_total = calc_reject(rj_tx, rj_dfe)

    return (
        "<H1>Jitter Rejection by Equalization Component</H1>"
        "<H2>Tx Preemphasis</H2>"
        '<TABLE border="1">'
        '<TR align="center">'
        "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>"
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">ISI</TD><TD>{isi_chnl:6.3f}</TD><TD>{isi_tx:6.3f}</TD><TD>{isi_rej_tx:4.1f}</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">DCD</TD><TD>{dcd_chnl:6.3f}</TD><TD>{dcd_tx:6.3f}</TD><TD>{dcd_rej_tx:4.1f}</TD>'
        '<TR align="right">'
        f'<TD align="center">Pj</TD><TD>{pj_chnl:6.3f}</TD><TD>{pj_tx:6.3f}</TD><TD>n/a</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Rj</TD><TD>{rj_chnl:6.3f}</TD><TD>{rj_tx:6.3f}</TD><TD>n/a</TD>'
        "</TR>"
        "</TABLE>"
        "<H2>CTLE</H2>"
        '<TABLE border="1">'
        '<TR align="center">'
        "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>"
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">ISI</TD><TD>{isi_tx:6.3f}</TD><TD>{isi_ctle:6.3f}</TD><TD>{isi_rej_ctle:4.1f}</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">DCD</TD><TD>{dcd_tx:6.3f}</TD><TD>{dcd_ctle:6.3f}</TD><TD>{dcd_rej_ctle:4.1f}</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Pj</TD><TD>{pj_tx:6.3f}</TD><TD>{pj_ctle:6.3f}</TD><TD>{pj_rej_ctle:4.1f}</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Rj</TD><TD>{rj_tx:6.3f}</TD><TD>{rj_ctle:6.3f}</TD><TD>{rj_rej_ctle:4.1f}</TD>'
        "</TR>"
        "</TABLE>"
        "<H2>DFE</H2>"
        '<TABLE border="1">'
        '<TR align="center">'
        "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>"
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">ISI</TD><TD>{isi_ctle:6.3f}</TD><TD>{isi_dfe:6.3f}</TD><TD>{isi_rej_dfe:4.1f}</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">DCD</TD><TD>{dcd_ctle:6.3f}</TD><TD>{dcd_dfe:6.3f}</TD><TD>{dcd_rej_dfe:4.1f}</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Pj</TD><TD>{pj_ctle:6.3f}</TD><TD>{pj_dfe:6.3f}</TD><TD>{pj_rej_dfe:4.1f}</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Rj</TD><TD>{rj_ctle:6.3f}</TD><TD>{rj_dfe:6.3f}</TD><TD>{rj_rej_dfe:4.1f}</TD>'
        "</TR>"
        "</TABLE>"
        "<H2>TOTAL</H2>"
        '<TABLE border="1">'
        '<TR align="center">'
        "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>"
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">ISI</TD><TD>{isi_chnl:6.3f}</TD><TD>{isi_dfe:6.3f}</TD><TD>{isi_rej_total:4.1f}</TD>'
        '<TR align="right">'
        f'<TD align="center">DCD</TD><TD>{dcd_chnl:6.3f}</TD><TD>{dcd_dfe:6.3f}</TD><TD>{dcd_rej_total:4.1f}</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Pj</TD><TD>{pj_tx:6.3f}</TD><TD>{pj_dfe:6.3f}</TD><TD>{pj_rej_total:4.1f}</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Rj</TD><TD>{rj_tx:6.3f}</TD><TD>{rj_dfe:6.3f}</TD><TD>{rj_rej_total:4.1f}</TD>'
        "</TR>"
        "</TABLE>"
    )


def calc_reject(num, dem):
    """Calculate rejection ratio or return 1.0e20."""
    return 10.0 * safe_log10(num / dem if dem else 1.0e20)


def performance_menu(perf):
    """Return the content for the performance tab of the GUI."""
    return (
        "<H2>Performance by Component</H2>"
        '<TABLE border="1">'
        '<TR align="center">'
        "<TH>Component</TH><TH>Performance (Msmpls./min.)</TH>"
        "</TR>"
        "<TR align=right>"
        f'<TD align="center">Channel</TD><TD>{perf["channel"]:6.3f}</TD>'
        "</TR>"
        "<TR align=right>"
        f'<TD align="center">Tx Preemphasis</TD><TD>{perf["tx"]:6.3f}</TD>'
        "</TR>"
        "<TR align=right>"
        f'<TD align="center">CTLE</TD><TD>{perf["ctle"]:6.3f}</TD>'
        "</TR>"
        "<TR align=right>"
        f'<TD align="center">DFE</TD><TD>{perf["dfe"]:6.3f}</TD>'
        "</TR>"
        "<TR align=right>"
        f'<TD align="center">Jitter Analysis</TD><TD>{perf["jitter"]:6.3f}</TD>'
        "</TR>"
        "<TR align=right>"
        f'<TD align="center"><strong>TOTAL</strong></TD><TD><strong>{perf["total"]:6.3f}</strong></TD>'
        "</TR>"
        "<TR align=right>"
        f'<TD align="center">Plotting</TD><TD>{perf["plot"]:6.3f}</TD>'
        "</TR>"
        "</TABLE>"
    )


def sweep_results_menu(sweep_results):
    """Return the content for the simulation sweep tab of the GUI."""
    sweep_table = [
        (
            f'<TR align="center">\n'
            f"<TD>{item[0]:06.3f}</TD>"
            f"<TD>{item[1]:06.3f}</TD>"
            f"<TD>{item[2]}</TD>"
            f"<TD>{item[3]}</TD>\n</TR>\n"
        )
        for item in sweep_results
    ]

    return (
        "<H2>Sweep Results</H2>"
        '<TABLE border="1">'
        '<TR align="center">'
        "<TH>Pretap</TH><TH>Posttap</TH><TH>Mean(bit errors)</TH><TH>StdDev(bit errors)</TH>"
        "</TR>"
        f"{sweep_table}"
        "</TABLE>"
    )


def status_string(status, total_perf, channel_delay, bit_errors, relative_power, jitter):
    """Return the status string across the bottom of the GUI."""
    status = (
        f"{status} | Perf. (Ms/m):    {total_perf:4.1}"
        f"         | ChnlDly (ns):    {channel_delay:5.3f}"
        f"         | BitErrs: {bit_errors}"
        f"         | TxPwr (W): {relative_power:4.2}"
    )
    try:
        jitter_status = (
            f"         | Jitter (ps):    "
            f"ISI={jitter.isi * 1.0e12:6.3f}    "
            f"DCD={jitter.dcd * 1.0e12:6.3f}    "
            f"Pj={jitter.pj * 1.0e12:6.3f}    "
            f"Rj={jitter.rj * 1.0e12:6.3f}"
        )
    except Exception:
        jitter_status = "         | (Jitter not available.)"

    return status + jitter_status
