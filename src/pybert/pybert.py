#! /usr/bin/env python

# pylint: disable=too-many-lines

"""
Bit error rate tester (BERT) simulator, written in Python.

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   17 June 2014

Testing by: Mark Marlett <mark.marlett@gmail.com>

This Python script provides a GUI interface to a BERT simulator, which
can be used to explore the concepts of serial communication link design.

Copyright (c) 2014 by David Banas; All rights reserved World wide.

ToDo:
    1. Add optional AFE (4th-order Bessel-Thomson).
    2. Add eye contour plots.
"""

import logging
import platform
import queue
import time
from datetime import datetime
from os.path import dirname, join
from pathlib import Path
from typing import Callable, Optional

import numpy as np  # type: ignore
import skrf as rf
from numpy import arange, array, cos, exp, pad, pi, sinc, where, zeros
from numpy.fft import irfft, rfft  # type: ignore
from numpy.random import randint  # type: ignore
from pyibisami import AMIModel, AMIParamConfigurator, IBISModel
from pyibisami import __version__ as PyAMI_VERSION  # type: ignore
from PySide6.QtCore import QObject, QTimer, Signal
from scipy.interpolate import interp1d

from pybert import __version__ as VERSION
from pybert.bert import SimulationThread
from pybert.configuration import Configuration, InvalidConfigFileType
from pybert.constants import gPeakFreq, gPeakMag
from pybert.models.stimulus import BitPattern, ModulationType
from pybert.models.tx_tap import TxTapTuner
from pybert.optimization import OptThread
from pybert.results import Results
from pybert.utility import (
    calc_gamma,
    import_channel,
    lfsr_bits,
    raised_cosine,
    safe_log10,
    sdd_21,
    trim_impulse,
)
from pybert.utility.jitter import JitterAnalysis

logger = logging.getLogger("pybert")


class PyBERT(QObject):  # pylint: disable=too-many-instance-attributes
    """A serial communication link bit error rate tester (BERT) simulator with a GUI interface.

    Useful for exploring the concepts of serial communication link design.
    """

    # QT Signals - Most are emitted from the `handle_results()` method
    sim_complete = Signal(object, object)  # Simulation complete signal, update all the plots
    opt_complete = Signal(object)  # Optimization complete signal, update the boost and plot
    opt_loop_complete = Signal(object)  # Optimization loop complete signal
    status_update = Signal(str)  # Tells the GUI to update the status bar but do not use the logger instance.

    def __init__(self, run_simulation: bool = False) -> None:
        """Initialize the PyBERT class.

        Args:
            run_simulation(Bool): If true, run the simulation, as part
                of class initialization. This is provided as an argument
                for the sake of larger applications, which may be
                importing PyBERT for its attributes and methods, and may
                not want to run the full simulation. (Optional;
                default = True)
        """
        super().__init__()

        # Independent variables

        # - Simulation Control
        self.bit_rate: float = 10.0  #: (Gbps)
        self.nbits: int = 15000  #: Number of bits to simulate.
        self.eye_bits: int = 10160  #: Number of bits used to form eye.
        self.pattern: BitPattern = BitPattern.PRBS7  #: Pattern to use for simulation.
        self.seed: int = 1  # LFSR seed. 0 means regenerate bits, using a new random seed, each run.
        self.nspui: int = 32  #: Signal vector samples per unit interval.
        self.mod_type: ModulationType = ModulationType.NRZ  #: 0 = NRZ; 1 = Duo-binary; 2 = PAM-4
        self.do_sweep: bool = False  #: Run sweeps? (Default = False)
        self.thresh: float = 3.0  #: Spectral threshold for identifying periodic components (sigma). (Default = 3.0)

        # - Channel Control
        self.channel_elements: list[dict[str, str]] = []  #: Channel elements.
        self.ch_file: str = ""  #: Channel file name.
        self.use_ch_file: bool = False  #: Import channel description from file? (Default = False)
        self.renumber: bool = False  #: Automatically fix "1=>3/2=>4" port numbering? (Default = False)
        self.f_step: float = 10  #: Frequency step to use when constructing H(f) (MHz). (Default = 10 MHz)
        self.f_max: float = 40  #: Frequency maximum to use when constructing H(f) (GHz). (Default = 40 GHz)
        self.impulse_length: float = 0.0  #: Impulse response length. (Determined automatically, when 0.)
        self.Rdc: float = 0.1876  #: Channel d.c. resistance (Ohms/m).
        self.w0: float = 10e6  #: Channel transition frequency (rads./s).
        self.R0: float = 1.452  #: Channel skin effect resistance (Ohms/m).
        self.Theta0: float = 0.02  #: Channel loss tangent (unitless).
        self.Z0: float = 100  #: Channel characteristic impedance, in LC region (Ohms).
        self.v0: float = 0.67  #: Channel relative propagation velocity (c).
        self.l_ch: float = 0.5  #: Channel length (m).
        self.use_window: bool = False  #: Apply raised cosine to frequency response before FFT()-ing? (Default = False)

        # - EQ Tune
        self.tx_tap_tuners: list[TxTapTuner] = [
            TxTapTuner(name="Pre-tap3", pos=-3, enabled=True, min_val=-0.05, max_val=0.05, step=0.025),
            TxTapTuner(name="Pre-tap2", pos=-2, enabled=True, min_val=-0.1, max_val=0.1, step=0.05),
            TxTapTuner(name="Pre-tap1", pos=-1, enabled=True, min_val=-0.2, max_val=0.2, step=0.1),
            TxTapTuner(name="Post-tap1", pos=1, enabled=True, min_val=-0.2, max_val=0.2, step=0.1),
            TxTapTuner(name="Post-tap2", pos=2, enabled=True, min_val=-0.1, max_val=0.1, step=0.05),
            TxTapTuner(name="Post-tap3", pos=3, enabled=True, min_val=-0.05, max_val=0.05, step=0.025),
        ]  #: EQ optimizer list of TxTapTuner objects.
        self.rx_bw_tune: float = 12.0  #: EQ optimizer CTLE bandwidth (GHz).
        self.peak_freq_tune: float = gPeakFreq  #: EQ optimizer CTLE peaking freq. (GHz).
        self.peak_mag_tune: float = gPeakMag  #: EQ optimizer CTLE peaking mag. (dB).
        self.min_mag_tune: float = 2  #: EQ optimizer CTLE peaking mag. min. (dB).
        self.max_mag_tune: float = 12  #: EQ optimizer CTLE peaking mag. max. (dB).
        self.step_mag_tune: float = 1  #: EQ optimizer CTLE peaking mag. step (dB).
        self.ctle_enable_tune: bool = True  #: EQ optimizer CTLE enable
        self.dfe_tap_tuners: list[TxTapTuner] = [
            TxTapTuner(name="Tap1", enabled=True, min_val=0.1, max_val=0.4, value=0.1),
            TxTapTuner(name="Tap2", enabled=True, min_val=-0.15, max_val=0.15, value=0.0),
            TxTapTuner(name="Tap3", enabled=True, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap4", enabled=True, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap5", enabled=True, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap6", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap7", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap8", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap9", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap10", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap11", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap12", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap13", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap14", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap15", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap16", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap17", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap18", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap19", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
            TxTapTuner(name="Tap20", enabled=False, min_val=-0.05, max_val=0.1, value=0.0),
        ]  #: EQ optimizer list of DFE tap tuner objects.

        # - Tx
        self.vod: float = 1.0  #: Tx differential output voltage (V)
        self.rs: int = 100  #: Tx source impedance (Ohms)
        self.cout: float = 0.5  #: Tx parasitic output capacitance (pF)
        self.pn_mag: float = 0.1  #: Periodic noise magnitude (V).
        self.pn_freq: float = 11  #: Periodic noise frequency (MHz).
        self.rn: float = 0.1  #: Standard deviation of Gaussian random noise (V).
        self.tx_taps: list[TxTapTuner] = [
            TxTapTuner(name="Pre-tap3", pos=-3, enabled=True, min_val=-0.05, max_val=0.05),
            TxTapTuner(name="Pre-tap2", pos=-2, enabled=True, min_val=-0.1, max_val=0.1),
            TxTapTuner(name="Pre-tap1", pos=-1, enabled=True, min_val=-0.2, max_val=0.2),
            TxTapTuner(name="Post-tap1", pos=1, enabled=True, min_val=-0.2, max_val=0.2),
            TxTapTuner(name="Post-tap2", pos=2, enabled=True, min_val=-0.1, max_val=0.1),
            TxTapTuner(name="Post-tap3", pos=3, enabled=True, min_val=-0.05, max_val=0.05),
        ]  #: List of TxTapTuner objects.
        self.rel_power: float = 1.0  #: Tx power dissipation (W).
        self.tx_use_ami: bool = False  #: (Bool)
        self.tx_has_ts4: bool = False  #: (Bool)
        self.tx_use_ts4: bool = False  #: (Bool)
        self.tx_use_getwave: bool = False  #: (Bool)
        self.tx_has_getwave: bool = False  #: (Bool)
        self.tx_ami_file: str = ""  #: (File)
        self.tx_ami_valid: bool = False  #: (Bool)
        self.tx_dll_file: str = ""  #: (File)
        self.tx_dll_valid: bool = False  #: (Bool)
        self.tx_ibis_file: str = ""  #: (File)
        self.tx_ibis_valid: bool = False  #: (Bool)
        self.tx_use_ibis: bool = False  #: (Bool)

        # - Rx
        self.rin: int = 100  #: Rx input impedance (Ohm)
        self.cin: float = 0.5  #: Rx parasitic input capacitance (pF)
        self.cac: float = 1.0  #: Rx a.c. coupling capacitance (uF)
        self.use_ctle_file: bool = False  #: For importing CTLE impulse/step response directly.
        self.ctle_file: str = ""  #: CTLE response file (when use_ctle_file = True).
        self.rx_bw: float = 12.0  #: CTLE bandwidth (GHz).
        self.peak_freq: float = gPeakFreq  #: CTLE peaking frequency (GHz)
        self.peak_mag: float = gPeakMag  #: CTLE peaking magnitude (dB)
        self.ctle_enable: bool = True  #: CTLE enable.
        self.rx_use_ami: bool = False  #: (Bool)
        self.rx_has_ts4: bool = False  #: (Bool)
        self.rx_use_ts4: bool = False  #: (Bool)
        self.rx_use_getwave: bool = False  #: (Bool)
        self.rx_has_getwave: bool = False  #: (Bool)
        self.rx_use_clocks: bool = False  #: (Bool)
        self.rx_ami_file: str = ""  #: (File)
        self.rx_ami_valid: bool = False  #: (Bool)
        self.rx_dll_file: str = ""  #: (File)
        self.rx_dll_valid: bool = False  #: (Bool)
        self.rx_ibis_file: str = ""  #: (File)
        self.rx_ibis_valid: bool = False  #: (Bool)
        self.rx_use_ibis: bool = False  #: (Bool)

        # - DFE
        self.sum_ideal: bool = True  #: True = use an ideal (i.e. - infinite bandwidth) summing node (Bool).
        self.decision_scaler: float = 0.5  #: DFE slicer output voltage (V).
        self.gain: float = 0.2  #: DFE error gain (unitless).
        self.n_ave: int = 100  #: DFE # of averages to take, before making tap corrections.
        self.sum_bw: float = 12.0  #: DFE summing node bandwidth (Used when sum_ideal=False.) (GHz).

        # - CDR
        self.delta_t: float = 0.1  #: CDR proportional branch magnitude (ps).
        self.alpha: float = 0.01  #: CDR integral branch magnitude (unitless).
        self.n_lock_ave: int = 500  #: CDR # of averages to take in determining lock.
        self.rel_lock_tol: float = 0.1  #: CDR relative tolerance to use in determining lock.
        self.lock_sustain: int = 500  #: CDR hysteresis to use in determining lock.

        # Misc.

        # Status
        self.len_h: float = 0
        self.chnl_dly: float = 0.0  #: Estimated channel delay (s).
        self.bit_errs: int = 0  #: # of bit errors observed in last run.
        self.run_count: int = 0  # Used as a mechanism to force bit stream regeneration.
        self.dfe_out_p: list = []

        self.tx_ibis = None
        self.tx_ibis_dir = ""
        self.tx_cfg = None
        self.tx_model = None
        self.rx_ibis = None
        self.rx_ibis_dir = ""
        self.rx_cfg = None
        self.rx_model = None

        # Initialize jitter analysis objects
        self.chnl_jitter: JitterAnalysis | None = None
        self.tx_jitter: JitterAnalysis | None = None
        self.ctle_jitter: JitterAnalysis | None = None
        self.dfe_jitter: JitterAnalysis | None = None

        # Threading and Processing
        self.simulation_thread: Optional[SimulationThread] = None  # Simulation Thread
        self.opt_thread: Optional[OptThread] = None  #: EQ optimization thread.

        # Setup a threading Queue to share results between threads
        self.result_queue: queue.Queue = queue.Queue()
        self.result_timer: QTimer = QTimer()
        self.result_timer.timeout.connect(self.poll_results)
        self.result_timer.start(100)  # Poll every 100 ms

        self.last_results: Results | None = None

        if run_simulation:
            self.simulate()

    # Dependent variable definitions
    @property
    def t(self):
        """Calculate the system time vector, in seconds."""

        ui = self.ui
        nspui = self.nspui
        nui = self.nui

        t0 = ui / nspui
        npts = nui * nspui

        return array([i * t0 for i in range(npts)])

    @property
    def t_ns(self):
        """Calculate the system time vector, in ns."""

        return self.t * 1.0e9

    @property
    def f(self):
        """
        Calculate the frequency vector for channel model construction.
        """
        fstep = self.f_step * 1e6
        fmax = self.f_max * 1e9
        return arange(0, fmax + fstep, fstep)  # "+fstep", so fmax gets included

    @property
    def w(self):
        """
        Channel modeling frequency vector, in rads./sec.
        """
        return 2 * pi * self.f

    @property
    def t_irfft(self):
        """
        Calculate the time vector appropriate for indexing `irfft()` output.
        """
        f = self.f
        tmax = 1 / f[1]
        tstep = 0.5 / f[-1]
        return arange(0, tmax, tstep)

    @property
    def bits(self):
        "Generate the bit stream."
        pattern = self.pattern.value
        seed = self.seed
        nbits = self.nbits

        if not seed:  # The user sets `seed` to zero when she wants a new random seed generated for each run.
            seed = randint(128)
            while not seed:  # We don't want to seed our LFSR with zero.
                seed = randint(128)
        bit_gen = lfsr_bits(pattern, seed)
        bits = [next(bit_gen) for _ in range(nbits)]
        return array(bits)

    @property
    def ui(self):
        """
        Returns the "unit interval" (i.e. - the nominal time span of each symbol moving through the channel).
        """

        mod_type = self.mod_type
        bit_rate = self.bit_rate * 1.0e9

        ui = 1.0 / bit_rate
        if mod_type == ModulationType.PAM4:  # PAM-4
            ui *= 2.0

        return ui

    @property
    def nui(self):
        """Returns the number of unit intervals in the test vectors."""

        mod_type = self.mod_type
        nbits = self.nbits

        nui = nbits
        if mod_type == ModulationType.PAM4:  # PAM-4
            nui //= 2

        return nui

    @property
    def eye_uis(self):
        """Returns the number of unit intervals to use for eye construction."""

        mod_type = self.mod_type
        eye_bits = self.eye_bits

        eye_uis = eye_bits
        if mod_type == ModulationType.PAM4:  # PAM-4
            eye_uis //= 2

        return eye_uis

    @property
    def ideal_h(self):
        """Returns the ideal link impulse response."""

        ui = self.ui.value
        nspui = self.nspui
        t = self.t
        mod_type = self.mod_type
        ideal_type = self.ideal_type[0]

        t = array(t) - t[-1] / 2.0

        if ideal_type == 0:  # delta
            ideal_h = zeros(len(t))
            ideal_h[len(t) / 2] = 1.0
        elif ideal_type == 1:  # sinc
            ideal_h = sinc(t / (ui / 2.0))
        elif ideal_type == 2:  # raised cosine
            ideal_h = (cos(pi * t / (ui / 2.0)) + 1.0) / 2.0
            ideal_h = where(t < -ui / 2.0, zeros(len(t)), ideal_h)
            ideal_h = where(t > ui / 2.0, zeros(len(t)), ideal_h)
        else:
            raise ValueError("PyBERT._get_ideal_h(): ERROR: Unrecognized ideal impulse response type.")

        if (
            mod_type == ModulationType.DUO
        ):  # Duo-binary relies upon the total link impulse response to perform the required addition.
            ideal_h = 0.5 * (ideal_h + pad(ideal_h[: -1 * nspui], (nspui, 0), "constant", constant_values=(0, 0)))

        return ideal_h

    @property
    def symbols(self):
        """Generate the symbol stream."""

        mod_type = self.mod_type
        vod = self.vod
        bits = self.bits

        if mod_type == ModulationType.NRZ:  # NRZ
            symbols = 2 * bits - 1
        elif mod_type == ModulationType.DUO:  # Duo-binary
            symbols = [bits[0]]
            for bit in bits[1:]:  # XOR pre-coding prevents infinite error propagation.
                symbols.append(bit ^ symbols[-1])
            symbols = 2 * array(symbols) - 1
        elif mod_type == ModulationType.PAM4:  # PAM-4
            symbols = []
            for bits in zip(bits[0::2], bits[1::2]):
                if bits == (0, 0):
                    symbols.append(-1.0)
                elif bits == (0, 1):
                    symbols.append(-1.0 / 3.0)
                elif bits == (1, 0):
                    symbols.append(1.0 / 3.0)
                else:
                    symbols.append(1.0)
        else:
            raise ValueError("ERROR: _get_symbols(): Unknown modulation type requested!")

        return array(symbols) * vod

    @property
    def ffe(self):
        """Generate the Tx pre-emphasis FIR numerator."""

        tap_tuners = self.tx_taps

        taps = []
        for tuner in tap_tuners:
            if tuner.enabled:
                taps.append(tuner.value)
            else:
                taps.append(0.0)
        curs_pos = -tap_tuners[0].pos
        curs_val = 1.0 - sum(abs(array(taps)))
        if curs_pos < 0:
            taps.insert(0, curs_val)
        else:
            taps.insert(curs_pos, curs_val)

        return taps

    def load_ibis_file(
        self,
        filepath: Path | str,
        is_tx: bool = True,
        current_component: str = None,
        current_pin: str = None,
        current_model: str = None,
    ) -> IBISModel:
        """Load a new IBIS file and return the IBISModel object."""
        try:
            logger.info("Parsing IBIS file: %s", str(filepath))
            # TODO: Some models are very large, we should lazy load them.
            ibis = IBISModel.from_file(filepath, is_tx=is_tx)
            if current_component and current_pin and current_model:
                ibis.current_component = current_component  # This updates the pin dictionary
                ibis.current_pin = current_pin  # This updates the model dictionary
                ibis.current_model = current_model  # Finally set the model.

            ibis_type = "tx" if is_tx else "rx"
            setattr(self, f"{ibis_type}_ibis", ibis)
            setattr(self, f"{ibis_type}_use_ibis", True)
            logger.info("Loaded new IBIS file. Switching to IBIS mode.")
            return ibis
        except Exception as err:  # pylint: disable=broad-exception-caught
            error_message = f"Failed to open and/or parse IBIS file!\n{err}"
            logger.exception(error_message)

    def load_ami_configurator(self, ami_file: Path | str, is_tx: bool = True) -> AMIParamConfigurator:
        """Load the AMI file and return the AMIParamConfigurator object."""
        try:
            logger.info("Parsing AMI file, '%s'...", str(ami_file))
            pcfg = AMIParamConfigurator.from_file(ami_file)
            ibis_type = "tx" if is_tx else "rx"
            # TODO: Move all of this into the AMIParamConfigurator class, pybert.py should just call the methods.
            if pcfg is not None:
                setattr(self, f"{ibis_type}_cfg", pcfg)
                setattr(self, f"{ibis_type}_has_ts4", pcfg.ts4file is not None)
                setattr(self, f"{ibis_type}_use_ami", True)
                logger.info("Loaded new Tx AMI file. Tx switching to AMI equalization mode.")
            else:
                logger.warning("Failed to load AMI file for Tx IBIS file. Tx will use native equalization.")
                setattr(self, f"{ibis_type}_use_ami", False)
            if pcfg.ami_parsing_errors:
                logger.warning(f"Non-fatal parsing errors:\n{pcfg.ami_parsing_errors}")
            else:
                logger.info("Success.")
            has_getwave = pcfg.getwave_exists()
            init_returns_impulse = pcfg.returns_impulse()
            if not init_returns_impulse:
                setattr(self, f"{ibis_type}_use_getwave", True)
            if pcfg.ts4file():
                has_ts4 = True
            else:
                has_ts4 = False
            setattr(self, f"{ibis_type}_has_ts4", has_ts4)
            setattr(self, f"{ibis_type}_use_ami", True)
            return pcfg
        except Exception as err:  # pylint: disable=broad-exception-caught
            error_message = f"Failed to open and/or parse AMI file!\n{err}"
            logger.exception(error_message)

    def load_dll_model(self, dll_file: Path | str, is_tx: bool = True):
        """Load the DLL/SO file and return the AMIModel object."""
        try:
            model = AMIModel(dll_file)
            ibis_type = "tx" if is_tx else "rx"
            setattr(self, f"{ibis_type}_model", model)
            return model
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to open DLL/SO file! %s", err)

    # This function has been pulled outside of the standard Traits/UI "depends_on / @property" mechanism,
    # in order to more tightly control when it executes. I wasn't able to get truly lazy evaluation, and
    # this was causing noticeable GUI slowdown.
    # pylint: disable=attribute-defined-outside-init
    def calc_chnl_h(self):
        """Calculates the channel impulse response.

        Also sets, in 'self':
         - chnl_dly:
             group delay of channel
         - start_ix:
             first element of trimmed response
         - t_ns_chnl:
             the x-values, in ns, for plotting 'chnl_h'
         - chnl_H:
             channel frequency response
         - chnl_s:
             channel step response
         - chnl_p:
             channel pulse response
        """

        t = self.t  # This time vector has NO relationship to `f`/`w`!
        t_irfft = self.t_irfft  # This time vector IS related to `f`/`w`.
        f = self.f
        w = self.w
        nspui = self.nspui
        impulse_length = self.impulse_length * 1.0e-9
        Rs = self.rs
        Cs = self.cout * 1.0e-12
        RL = self.rin
        Cp = self.cin * 1.0e-12
        # CL = self.cac * 1.0e-6  # pylint: disable=unused-variable

        ts = t[1]
        len_f = len(f)

        # Form the pre-on-die S-parameter 2-port network for the channel.
        if self.use_ch_file:
            # TODO: This is temporary until we support multiple channel files.
            self.ch_file = self.channel_elements[0][0]
            ch_s2p_pre = import_channel(self.ch_file, ts, f, renumber=self.renumber)
            logger.info(str(ch_s2p_pre))
            H = ch_s2p_pre.s21.s.flatten()
        else:
            # Construct PyBERT default channel model (i.e. - Howard Johnson's UTP model).
            # - Grab model parameters from PyBERT instance.
            l_ch = self.l_ch
            v0 = self.v0 * 3.0e8
            R0 = self.R0
            w0 = self.w0
            Rdc = self.Rdc
            Z0 = self.Z0
            Theta0 = self.Theta0
            # - Calculate propagation constant, characteristic impedance, and transfer function.
            gamma, Zc = calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, w)
            self.Zc = Zc
            H = exp(-l_ch * gamma)  # pylint: disable=invalid-unary-operand-type
            self.H = H
            # - Use the transfer function and characteristic impedance to form "perfectly matched" network.
            tmp = np.array(list(zip(zip(zeros(len_f), H), zip(H, zeros(len_f)))))
            ch_s2p_pre = rf.Network(s=tmp, f=f / 1e9, z0=Zc)
            # - And, finally, renormalize to driver impedance.
            ch_s2p_pre.renormalize(Rs)
        try:
            ch_s2p_pre.name = "ch_s2p_pre"
        except Exception:  # pylint: disable=broad-exception-caught
            logger.info(f"ch_s2p_pre: {ch_s2p_pre}")
            raise
        self.ch_s2p_pre = ch_s2p_pre
        ch_s2p = ch_s2p_pre  # In case neither set of on-die S-parameters is being invoked, below.

        # Augment w/ IBIS-AMI on-die S-parameters, if appropriate.
        def add_ondie_s(s2p, ts4f, isRx=False):
            """Add the effect of on-die S-parameters to channel network.

            Args:
                s2p(skrf.Network): initial 2-port network.
                ts4f(string): on-die S-parameter file name.

            Keyword Args:
                isRx(bool): True when Rx on-die S-params. are being added. (Default = False).

            Returns:
                skrf.Network: Resultant 2-port network.
            """
            ts4N = rf.Network(ts4f)  # Grab the 4-port single-ended on-die network.
            ntwk = sdd_21(ts4N)  # Convert it to a differential, 2-port network.
            # Interpolate to system freqs.
            ntwk2 = (
                ntwk.extrapolate_to_dc()
                .windowed(normalize=False)
                .interpolate(s2p.f, coords="polar", bounds_error=False, fill_value="extrapolate")
            )
            if isRx:
                res = s2p**ntwk2
            else:  # Tx
                res = ntwk2**s2p
            return (res, ts4N, ntwk2)

        if self.tx_use_ibis:
            model = self.tx_ibis.current_model
            Rs = model.impedance * 2
            Cs = model.ccomp[0] / 2  # They're in series.
            self.Rs = Rs  # Primarily for debugging.
            self.Cs = Cs
            if self.tx_use_ts4:
                fname = join(self.tx_ibis_dir, self._tx_cfg.fetch_param_val(["Reserved_Parameters", "Ts4file"]))
                ch_s2p, ts4N, ntwk = add_ondie_s(ch_s2p, fname)
                self.ts4N = ts4N
                self.ntwk = ntwk
        if self.rx_use_ibis:
            model = self.rx_ibis.current_model
            RL = model.impedance * 2
            Cp = model.ccomp[0] / 2
            self.RL = RL  # Primarily for debugging.
            self.Cp = Cp
            logger.debug(f"RL: {round(RL, 2)}, Cp: {round(Cp, 2)}")
            if self.rx_use_ts4:
                fname = join(self.rx_ibis_dir, self._rx_cfg.fetch_param_val(["Reserved_Parameters", "Ts4file"]))
                ch_s2p, ts4N, ntwk = add_ondie_s(ch_s2p, fname, isRx=True)
                self.ts4N = ts4N
                self.ntwk = ntwk
        ch_s2p.name = "ch_s2p"
        self.ch_s2p = ch_s2p

        # Calculate channel impulse response.
        Zs = Rs / (1 + 1j * w * Rs * Cs)  # Tx termination impedance
        Zt = RL / (1 + 1j * w * RL * Cp)  # Rx termination impedance
        ch_s2p_term = ch_s2p.copy()
        ch_s2p_term_z0 = ch_s2p.z0.copy()
        ch_s2p_term_z0[:, 0] = Zs
        ch_s2p_term_z0[:, 1] = Zt
        ch_s2p_term.renormalize(ch_s2p_term_z0)
        ch_s2p_term.name = "ch_s2p_term"
        self.ch_s2p_term = ch_s2p_term

        # We take the transfer function, H, to be a ratio of voltages.
        # So, we must normalize our (now generalized) S-parameters.
        chnl_H = ch_s2p_term.s21.s.flatten() * np.sqrt(ch_s2p_term.z0[:, 1] / ch_s2p_term.z0[:, 0])
        if self.use_window:
            chnl_h = irfft(raised_cosine(chnl_H))
        else:
            chnl_h = irfft(chnl_H)
        krnl = interp1d(t_irfft, chnl_h, kind="cubic", bounds_error=False, fill_value=0, assume_sorted=True)
        temp = krnl(t)
        chnl_h = temp * t[1] / t_irfft[1]
        chnl_dly = where(chnl_h == max(chnl_h))[0][0] * ts

        min_len = 20 * nspui
        max_len = 100 * nspui
        if impulse_length:
            min_len = max_len = int(impulse_length / ts)
        chnl_h, start_ix = trim_impulse(chnl_h, min_len=min_len, max_len=max_len, front_porch=True, kept_energy=0.999)
        krnl = interp1d(t[: len(chnl_h)], chnl_h, kind="cubic", bounds_error=False, fill_value=0, assume_sorted=True)
        chnl_trimmed_H = rfft(krnl(t_irfft)) * t_irfft[1] / t[1]

        chnl_s = chnl_h.cumsum()
        chnl_p = chnl_s - pad(
            chnl_s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0)
        )  # pylint: disable=invalid-unary-operand-type

        self.chnl_h = chnl_h
        self.len_h = len(chnl_h)
        self.chnl_dly = chnl_dly
        self.chnl_H = chnl_H
        self.chnl_H_raw = H
        self.chnl_trimmed_H = chnl_trimmed_H
        self.start_ix = start_ix
        self.t_ns_chnl = array(t[start_ix : start_ix + len(chnl_h)]) * 1.0e9
        self.chnl_s = chnl_s
        self.chnl_p = chnl_p

        return chnl_h

    def load_configuration(self, filepath: Path | str):
        """Load in a configuration into pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            Configuration.load_from_file(filepath, self)
            logger.info("Loaded configuration.")
        except InvalidConfigFileType:
            logger.error("This filetype is not currently supported.")
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load configuration. See the console for more detail.")
            logger.exception(str(err))

    def save_configuration(self, filepath: Path | str):
        """Save out a configuration from pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            Configuration(self).save(filepath)
            logger.info("Configuration saved.")
        except InvalidConfigFileType:
            logger.error("This filetype is not currently supported. Please try again as a yaml file.")
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error(f"Failed to save configuration:\n\t{err}")

    def reset_configuration(self) -> None:
        """Reset the PyBERT instance to default configuration values."""
        Configuration.apply_default_config(self)
        logger.info("Default configuration applied.")

    def load_results(self, filepath: Path) -> Results:
        """Load results from a file into pybert.

        Args:
            filepath: A full filepath include the suffix.
        """
        try:
            self.last_results = Results.load_from_file(filepath)
            logger.info("Loaded results.")
            return self.last_results
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load results from file. See the console for more detail.")
            logger.exception(str(err))

    def save_results(self, filepath: Path):
        """Save the existing results to a pickle file.

        Args:
            filepath: A full filepath include the suffix.
        """
        if self.last_results:
            try:
                Results(results=self.last_results["results"], performance=self.last_results["performance"]).save(
                    filepath
                )
                logger.info("Saved results.")
            except Exception as err:  # pylint: disable=broad-exception-caught
                logger.error("Failed to save results to file. See the console for more detail.")
                logger.exception(str(err))
        else:
            logger.error("No results to save. Please run a simulation first.")

    def simulate(self, wait_for_completion: bool = False):
        """Start a simulation of the current configuration in a separate thread."""
        if self.simulation_thread and self.simulation_thread.is_alive():
            pass
        elif self.is_valid_configuration():
            logger.info("Starting simulation.")
            self.simulation_thread = SimulationThread()
            self.simulation_thread.pybert = self
            self.simulation_thread.start()
            if wait_for_completion:
                self.simulation_thread.join()

    def stop_simulation(self):
        """Stop the running simulation."""
        logger.info("Stopping simulation.")
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.stop()
            self.simulation_thread.join(10)

    def calculate_optimization_trials(self):
        """Calculate the number of trials for the optimization."""
        n_trials = int((self.max_mag_tune - self.min_mag_tune) / self.step_mag_tune)
        for tuner in self.tx_tap_tuners:
            n_trials *= int((tuner.max_val - tuner.min_val) / tuner.step)
        return n_trials

    def optimize(self):
        """Start the optimization process using the tuner values.

        If the user accidently sets a large number of trials, prompt them to confirm before proceeding.
        """
        if self.opt_thread and self.opt_thread.is_alive():
            pass
        elif self.is_valid_configuration():
            logger.info("Starting optimization.")
            self.opt_thread = OptThread()
            self.opt_thread.pybert = self
            self.opt_thread.start()

    def stop_optimization(self):
        """Stop the running optimization."""
        logger.info("Stopping optimization.")
        if self.opt_thread and self.opt_thread.is_alive():
            self.opt_thread.stop()
            self.opt_thread.join(10)

    def reset_optimization(self):
        """Reset the optimization back to what the current configuration is."""
        logger.info("Resetting optimization.")
        for i, tap in enumerate(self.tx_taps):
            self.tx_tap_tuners[i].value = tap.value
            self.tx_tap_tuners[i].enabled = tap.enabled
        self.peak_freq_tune = self.peak_freq
        self.peak_mag_tune = self.peak_mag
        self.rx_bw_tune = self.rx_bw
        self.ctle_enable_tune = self.ctle_enable

    def apply_optimization(self):
        """Apply the optimization to the current configuration."""
        logger.info("Applying optimization.")
        for i, tap in enumerate(self.tx_tap_tuners):
            self.tx_taps[i].value = tap.value
            self.tx_taps[i].enabled = tap.enabled
        self.peak_freq = self.peak_freq_tune
        self.peak_mag = self.peak_mag_tune
        self.rx_bw = self.rx_bw_tune
        self.ctle_enable = self.ctle_enable_tune

    def is_valid_configuration(self):
        """Validate that the user has selected a valid configuration for simulation or optimization."""
        if not self.channel_elements and self.use_ch_file:
            logger.error("No channel file selected. Please select a channel file.")
            return False
        if not self.tx_ibis and self.tx_use_ibis:
            logger.error("No Tx IBIS file selected. Please select a Tx IBIS file.")
            return False
        if not self.rx_ibis and self.rx_use_ibis:
            logger.error("No Rx IBIS file selected. Please select a Rx IBIS file.")
            return False
        if not self.tx_cfg and self.tx_model and self.tx_use_ami:
            logger.error("No Tx AMI loaded or configured.")
            return False
        if not self.rx_cfg and self.rx_model and self.rx_use_ami:
            logger.error("No Tx AMI loaded or configured.")
            return False
        return True

    def poll_results(self):
        """The Qt timer calls this function every 100ms to check if there are any results in the queue."""
        try:
            while True:
                result = self.result_queue.get_nowait()
                self.handle_result(result)
        except queue.Empty:
            pass

    def handle_result(self, result):
        """Handle the results from from the thread message queue.

        The QT framework expects that all signal/slots that update the GUI are called from the main thread. This function
        allows us to communicate between the threads without breaking this requirement.  Otherwise, the GUI would freeze
        and error out with a QTimer timing out from trying to keep the main thread updated.

        Args:
            result: The result from the thread message queue.
        """
        # The optimizer will put the end results in the queue if complete and valid.
        if result.get("type") == "optimization":
            tx_weights = result.get("tx_weights", [])
            rx_peaking = result.get("rx_peaking", 0)
            fom = result.get("fom", 0)
            for k, tx_weight in enumerate(tx_weights):
                self.tx_tap_tuners[k].value = tx_weight
            self.peak_mag_tune = rx_peaking
            self.opt_complete.emit(self.peak_mag_tune)
        # This is used to update the plots during the optimization loop.
        elif result.get("type") == "opt_loop_complete":
            self.opt_loop_complete.emit(result)
        # This is used to update only the status bar this is for something that has a progress or a lot of updates
        elif result.get("type") == "status_update":
            self.status_update.emit(result.get("message"))
        # This is used to update the plots and the results after a simulation is complete.
        elif result.get("type") == "simulation_complete":
            self.last_results = result
            self.sim_complete.emit(result["results"], result["performance"])
