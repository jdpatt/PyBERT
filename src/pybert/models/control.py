import logging

from traits.api import Bool, Float, HasTraits, Int, List, Map, Range

from pybert.gui.control import CONTROL_VIEW

logger = logging.getLogger(__name__)

gDebugStatus = False
gMaxCTLEPeak = 20.0  # max. allowed CTLE peaking (dB) (when optimizing, only)
gMaxCTLEFreq = 20.0  # max. allowed CTLE peak frequency (GHz) (when optimizing, only)

# Default model parameters - Modify these to customize the default simulation.
# - Simulation Control
gBitRate = 10  # (Gbps)
gNbits = 8000  # number of bits to run
gPatLen = 127  # repeating bit pattern length
gNspb = 32  # samples per bit
gNumAve = 1  # Number of bit error samples to average, when sweeping.

# - Tx
gRn = (
    0.001  # standard deviation of Gaussian random noise (V) (Applied at end of channel, so as to appear white to Rx.)
)

gVod = 1.0  # output drive strength (Vp)
gPnMag = 0.001  # magnitude of periodic noise (V)
gPnFreq = 0.437  # frequency of periodic noise (MHz)
gPnMag = 0.001  # magnitude of periodic noise (V)
gPnFreq = 0.437  # frequency of periodic noise (MHz)

gPeakFreq = 5.0  # CTLE peaking frequency (GHz)
gPeakMag = 1.7  # CTLE peaking magnitude (dB)

gBW = 12.0  # Rx signal path bandwidth, assuming no CTLE action. (GHz)
gUseDfe = True  # Include DFE when running simulation.
gCTLEOffset = 0.0  # CTLE d.c. offset (dB)
gNtaps = 5

# - Analysis
gThresh = 6  # threshold for identifying periodic jitter spectral elements (sigma)


class Control(HasTraits):
    bit_rate = Range(low=0.1, high=120.0, value=gBitRate)  #: (Gbps)
    nbits = Range(low=1000, high=10000000, value=gNbits)  #: Number of bits to simulate.
    pattern = Map(
        {
            "PRBS-7": [7, 6],
            "PRBS-15": [15, 14],
            "PRBS-23": [23, 18],
        },
        default_value="PRBS-7",
    )
    seed = Int(1)  # LFSR seed. 0 means regenerate bits, using a new random seed, each run.
    nspb = Range(low=2, high=256, value=gNspb)  #: Signal vector samples per bit.
    eye_bits = Int(gNbits // 5)  #: # of bits used to form eye. (Default = last 20%)
    mod_type = List([0])  #: 0 = NRZ; 1 = Duo-binary; 2 = PAM-4
    num_sweeps = Int(1)  #: Number of sweeps to run.
    sweep_num = Int(1)
    sweep_aves = Int(gNumAve)
    do_sweep = Bool(False)  #: Run sweeps? (Default = False)
    impulse_length = Float(0.0)  #: Impulse response length. (Determined automatically, when 0.)
    vod = Float(gVod)  #: Tx differential output voltage (V)
    rn = Float(gRn)  #: Standard deviation of Gaussian random noise (V).
    pn_mag = Float(gPnMag)  #: Periodic noise magnitude (V).
    pn_freq = Float(gPnFreq)  #: Periodic noise frequency (MHz).
    # - Analysis
    thresh = Int(gThresh)  #: Threshold for identifying periodic jitter components (sigma).

    def default_traits_view(self):
        return CONTROL_VIEW

    def check_pat_len(self):
        taps = self.pattern_
        pat_len = 2 * pow(2, max(taps))
        if pat_len > 5 * self.nbits:
            logger.error(
                "Accurate jitter decomposition may not be possible with the current configuration!\n \
Try to keep Nbits & EyeBits > 10 * 2^n, where `n` comes from `PRBS-n`.",
                # alert=True,
            )

    def _pattern_changed(self, new_value):
        self.check_pat_len()

    def _nbits_changed(self, new_value):
        self.check_pat_len()
