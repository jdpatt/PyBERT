from logging import getLogger

from traits.api import (
    Bool,
    Enum,
    File,
    Float,
    List,
    Range,
)


# - Tx
gVod = 1.0  # output drive strength (Vp)
gRs = 100  # differential source impedance (Ohms)
gCout = 0.50  # parasitic output capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
gPnMag = 0.001  # magnitude of periodic noise (V)
gPnFreq = 0.437  # frequency of periodic noise (MHz)

# - Rx
gRin = 100  # differential input resistance
gCin = 0.50  # parasitic input capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
gCac = 1.0  # a.c. coupling capacitance (uF) (Assumed to exist at both 'P' and 'N' nodes.)
gBW = 12.0  # Rx signal path bandwidth, assuming no CTLE action. (GHz)
gUseDfe = True  # Include DFE when running simulation.
gDfeIdeal = True  # DFE ideal summing node selector
gPeakFreq = 5.0  # CTLE peaking frequency (GHz)
gPeakMag = 10.0  # CTLE peaking magnitude (dB)
gCTLEOffset = 0.0  # CTLE d.c. offset (dB)


class Buffer(object):
    """docstring for Buffer"""
    def __init__(self):
        super(Buffer, self).__init__()
        self.log = logging.getLogger("pybert.buffer")
        self.log.debug("Creating Buffer Object")
        self.use_ami = Bool(False)  #: (Bool)
        self.use_getwave = Bool(False)  #: (Bool)
        self.has_getwave = Bool(False)  #: (Bool)
        self.ami_file = File("", entries=5, filter=["*.ami"])  #: (File)
        self.ami_valid = Bool(False)  #: (Bool)
        self.dll_file = File("", entries=5, filter=["*.dll", "*.so"])  #: (File)
        self.dll_valid = Bool(False)  #: (Bool)
        self.open_gui = None

    def ami_file_changed(self, new_value):
        try:
            self.ami_valid = False
            with open(new_value) as pfile:
                pcfg = AMIParamConfigurator(pfile.read())
            self.log.info("Parsing AMI file, '{}'...\n{}".format(new_value, pcfg.ami_parsing_errors))
            self.has_getwave = pcfg.fetch_param_val(["Reserved_Parameters", "GetWave_Exists"])
            self.open_gui = pcfg.open_gui
            self.ami_valid = True
        except Exception as err:
            self.handle_error("Failed to open and/or parse AMI file!\n{}".format(err))

    def dll_file_changed(self, new_value):
        try:
            self.dll_valid = False
            model = AMIModel(str(new_value))
            self._model = model
            self.dll_valid = True
        except Exception as err:
            self.handle_error("Failed to open DLL/SO file!\n{}".format(err))


class TxTapTuner(HasTraits):
    """Object used to populate the rows of the Tx FFE tap tuning table."""

    def __init__(self,
            name: str = "(noname)",
            enabled: bool = False,
            min_val: float = 0.0,
            max_val: float = 0.0,
            value: float = 0.0,
            steps: int = 0
        ):
        """Allows user to define properties, at instantiation."""

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super(TxTapTuner, self).__init__()

        self.name = String(name)
        self.enabled = Bool(enabled)
        self.min_val = Float(min_val)
        self.max_val = Float(max_val)
        self.value = Float(value)
        self.steps = Int(steps)


class Transmitter(Buffer):
    """docstring for Transmitter"""
    def __init__(self):
        super(Transmitter, self).__init__()
        self.log = logging.getLogger("pybert.buffer.tx")
        self.log.debug("Creating Tx Object")
        self.vod = Float(gVod)  #: Tx differential output voltage (V)
        self.rs = Float(gRs)  #: Tx source impedance (Ohms)
        self.cout = Range(low=0.001, value=gCout)  #: Tx parasitic output capacitance (pF)
        self.pn_mag = Float(gPnMag)  #: Periodic noise magnitude (V).
        self.pn_freq = Float(gPnFreq)  #: Periodic noise frequency (MHz).
        self.rn = Float(gRn)  #: Standard deviation of Gaussian random noise (V).
        self.tx_taps = List(
            [
                TxTapTuner(name="Pre-tap", enabled=True, min_val=-0.2, max_val=0.2, value=0.0),
                TxTapTuner(name="Post-tap1", enabled=False, min_val=-0.4, max_val=0.4, value=0.0),
                TxTapTuner(name="Post-tap2", enabled=False, min_val=-0.3, max_val=0.3, value=0.0),
                TxTapTuner(name="Post-tap3", enabled=False, min_val=-0.2, max_val=0.2, value=0.0),
            ]
        )  #: List of TxTapTuner objects.
        self.rel_power = Float(1.0)  #: Tx power dissipation (W).


class Receiver(Buffer):
    """docstring for Receiver"""
    def __init__(self):
        super(Receiver, self).__init__()
        self.log = logging.getLogger("pybert.buffer.rx")
        self.log.debug("Creating Rx Object")
        self.rin = Float(gRin)  #: Rx input impedance (Ohm)
        self.cin = Range(low=0.001, value=gCin)  #: Rx parasitic input capacitance (pF)
        self.cac = Float(gCac)  #: Rx a.c. coupling capacitance (uF)
        self.use_ctle_file = Bool(False)  #: For importing CTLE impulse/step response directly.
        self.ctle_file = File("", entries=5, filter=["*.csv"])  #: CTLE response file (when use_ctle_file = True).
        self.rx_bw = Float(gBW)  #: CTLE bandwidth (GHz).
        self.peak_freq = Float(gPeakFreq)  #: CTLE peaking frequency (GHz)
        self.peak_mag = Float(gPeakMag)  #: CTLE peaking magnitude (dB)
        self.ctle_offset = Float(gCTLEOffset)  #: CTLE d.c. offset (dB)
        self.ctle_mode = Enum("Off", "Passive", "AGC", "Manual")  #: CTLE mode ('Off', 'Passive', 'AGC', 'Manual').
