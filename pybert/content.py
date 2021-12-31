"""
Static or generated content for PyBERT.

Original author: David Banas <capn.freako@gmail.com>

Original date:   April 15, 2015 (Copied from pybert.py.)

Copyright (c) 2015 David Banas; all rights reserved World wide.
"""


help_str = """<H2>PyBERT User's Guide</H2>\n
  <H3>Note to developers</H3>\n
    This is NOT for you. Instead, open 'pybert/doc/_build/html/index.html' in a browser.\n
  <H3>PyBERT User Help Options</H3>\n
    <UL>\n
      <LI>Hover over any user-settable value in the <em>Config.</em> tab, for help message.</LI>\n
      <LI>Peruse the <em>General Tips</em> & <em>Help by Tab</em> sections, below.</LI>\n
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


def performance_info_table(channel, tx, ctle, dfe, jitter, plotting, total) -> str:
    """Return a string html table of the performance of each element from the last run."""
    info_str = (
        "<H2>Performance by Component</H2>\n"
        '<TABLE border="1">\n'
        '  <TR align="center">\n'
        "    <TH>Component</TH><TH>Performance (Msmpls./min.)</TH>\n"
        '  </TR>\n  <TR align="right">\n'
    )
    info_str += f'    <TD align="center">Channel</TD><TD>{channel}</TD>\n'
    info_str += '  </TR>\n  <TR align="right">\n'
    info_str += f'    <TD align="center">Tx Preemphasis</TD><TD>{tx}</TD>\n'
    info_str += '  </TR>\n  <TR align="right">\n'
    info_str += f'    <TD align="center">CTLE</TD><TD>{ctle}</TD>\n'
    info_str += '  </TR>\n  <TR align="right">\n'
    info_str += f'    <TD align="center">DFE</TD><TD>{dfe}</TD>\n'
    info_str += '  </TR>\n  <TR align="right">\n'
    info_str += f'    <TD align="center">Jitter Analysis</TD><TD>{jitter}</TD>\n'
    info_str += '  </TR>\n  <TR align="right">\n'
    info_str += f'    <TD align="center">Plotting</TD><TD>{plotting}</TD>\n'
    info_str += '  </TR>\n  <TR align="right">\n'
    info_str += '    <TD align="center"><strong>TOTAL</strong></TD><TD><strong>%6.3f</strong></TD>\n' % (total)
    info_str += "  </TR>\n</TABLE>\n"

    return info_str


def sweep_info_table(sweep_results) -> str:
    """Return a string that is the summary of all sweep configurations and averages that ran."""
    info_str = (
        "<H2>Sweep Results</H2>\n"
        '<TABLE border="1">\n'
        '  <TR align="center">\n'
        "    <TH>Pretap</TH><TH>Posttap</TH><TH>Mean(bit errors)</TH><TH>StdDev(bit errors)</TH>\n"
        "  </TR>\n"
    )

    if sweep_results:
        for settings, bit_error_mean, bit_error_std in sweep_results:
            info_str += '  <TR align="center">\n'
            info_str += (
                f"    <TD>{settings[0]}</TD><TD>{settings[1:]}</TD><TD>{bit_error_mean}</TD><TD>{bit_error_std}</TD>\n"
            )
            info_str += "  </TR>\n"

    info_str += "</TABLE>\n"
    return info_str


def jitter_info_table(jitter) -> str:
    # Temporary, until I figure out DPI independence.
    info_str = "<style>\n"
    # info_str += ' table td {font-size: 36px;}\n'
    # info_str += ' table th {font-size: 38px;}\n'
    info_str += " table td {font-size: 12em;}\n"
    info_str += " table th {font-size: 14em;}\n"
    info_str += "</style>\n"
    # info_str += '<font size="+3">\n'
    # End Temp.

    info_str = "<H1>Jitter Rejection by Equalization Component</H1>\n"

    info_str += "<H2>Tx Preemphasis</H2>\n"
    info_str += '<TABLE border="1">\n'
    info_str += '<TR align="center">\n'
    info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["isi_chnl"],
        jitter["isi_tx"],
        jitter["isi_rej_tx"],
    )
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["dcd_chnl"],
        jitter["dcd_tx"],
        jitter["dcd_rej_tx"],
    )
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += (
        f'<TD align="center">Pj</TD><TD>{jitter["pj_chnl"]:6.3f}</TD><TD>{jitter["pj_tx"]:6.3f}</TD><TD>n/a</TD>\n'
    )
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += (
        f'<TD align="center">Rj</TD><TD>{jitter["rj_chnl"]:6.3f}</TD><TD>{jitter["rj_tx"]:6.3f}</TD><TD>n/a</TD>\n'
    )
    info_str += "</TR>\n"
    info_str += "</TABLE>\n"

    info_str += "<H2>CTLE (+ AMI DFE)</H2>\n"
    info_str += '<TABLE border="1">\n'
    info_str += '<TR align="center">\n'
    info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["isi_tx"],
        jitter["isi_ctle"],
        jitter["isi_rej_ctle"],
    )
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["dcd_tx"],
        jitter["dcd_ctle"],
        jitter["dcd_rej_ctle"],
    )
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["pj_tx"],
        jitter["pj_ctle"],
        jitter["pj_rej_ctle"],
    )
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["rj_tx"],
        jitter["rj_ctle"],
        jitter["rj_rej_ctle"],
    )
    info_str += "</TR>\n"
    info_str += "</TABLE>\n"

    info_str += "<H2>DFE</H2>\n"
    info_str += '<TABLE border="1">\n'
    info_str += '<TR align="center">\n'
    info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["isi_ctle"],
        jitter["isi_dfe"],
        jitter["isi_rej_dfe"],
    )
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["dcd_ctle"],
        jitter["dcd_dfe"],
        jitter["dcd_rej_dfe"],
    )
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["pj_ctle"],
        jitter["pj_dfe"],
        jitter["pj_rej_dfe"],
    )
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["rj_ctle"],
        jitter["rj_dfe"],
        jitter["rj_rej_dfe"],
    )
    info_str += "</TR>\n"
    info_str += "</TABLE>\n"

    info_str += "<H2>TOTAL</H2>\n"
    info_str += '<TABLE border="1">\n'
    info_str += '<TR align="center">\n'
    info_str += "<TH>Jitter Component</TH><TH>Input (ps)</TH><TH>Output (ps)</TH><TH>Rejection (dB)</TH>\n"
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">ISI</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["isi_chnl"],
        jitter["isi_dfe"],
        jitter["isi_rej_total"],
    )
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">DCD</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["dcd_chnl"],
        jitter["dcd_dfe"],
        jitter["dcd_rej_total"],
    )
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">Pj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["pj_tx"],
        jitter["pj_dfe"],
        jitter["pj_rej_total"],
    )
    info_str += "</TR>\n"
    info_str += '<TR align="right">\n'
    info_str += '<TD align="center">Rj</TD><TD>%6.3f</TD><TD>%6.3f</TD><TD>%4.1f</TD>\n' % (
        jitter["rj_tx"],
        jitter["rj_dfe"],
        jitter["rj_rej_total"],
    )
    info_str += "</TR>\n"
    info_str += "</TABLE>\n"

    return info_str


def jiffer_status_string(pybert) -> str:
    """Return the jitter portion of the statusbar string."""
    try:
        jit_str = "         | Jitter (ps):    ISI=%6.3f    DCD=%6.3f    Pj=%6.3f    Rj=%6.3f" % (
            pybert.isi_dfe * 1.0e12,
            pybert.dcd_dfe * 1.0e12,
            pybert.pj_dfe * 1.0e12,
            pybert.rj_dfe * 1.0e12,
        )
    except:
        jit_str = "         | (Jitter not available.)"
    return jit_str


def status_string(pybert) -> str:
    """Return a string for the status bar across the bottom with the following info:
    - simulation status
    - overall performance
    - channel delay
    - bit errors
    - tx power consumption
    - total jitter
    """
    status_str = "%-20s | Perf. (Msmpls./min.):  %4.1f" % (
        pybert.status,
        pybert.total_perf * 60.0e-6,
    )
    dly_str = f"         | ChnlDly (ns):    {pybert.chnl_dly * 1000000000.0:5.3f}"
    err_str = f"         | BitErrs: {pybert.bit_errs}"
    pwr_str = f"         | TxPwr (W): {pybert.rel_power:4.2f}"
    status_str += dly_str + err_str + pwr_str
    return status_str + jiffer_status_string(pybert)
