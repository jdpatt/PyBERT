"""
Any static content that gets rendered in the GUI.

Original author: David Banas <capn.freako@gmail.com>"

Original date:   April 15, 2015 (Copied from pybert.py.)

Copyright (c) 2015 David Banas; all rights reserved World wide.
"""


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


def jitter_rejection_menu():
    """Return the content for the jitter rejection tab of the GUI."""
    return (
        "<H1>Jitter Rejection by Equalization Component</H1>"
        "<H2>Tx Preemphasis</H2>"
        '<TABLE border="1">'
        '<TR align="center">'
        "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>"
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        '<TR align="right">'
        f'<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>n/a</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>n/a</TD>'
        "</TR>"
        "</TABLE>"
        "<H2>CTLE</H2>"
        '<TABLE border="1">'
        '<TR align="center">'
        "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>"
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        "</TR>"
        "</TABLE>"
        "<H2>DFE</H2>"
        '<TABLE border="1">'
        '<TR align="center">'
        "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>"
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        "</TR>"
        "</TABLE>"
        "<H2>TOTAL</H2>"
        '<TABLE border="1">'
        '<TR align="center">'
        "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>"
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        '<TR align="right">'
        f'<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        "</TR>"
        '<TR align="right">'
        f'<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>'
        "</TR>"
        "</TABLE>"
    )


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
