"""This file holds all the default settings that get populated in pybert."""

DEBUG = True
DEBUG_OPTIMIZE = False
MAX_CTLE_PEAK = 20.0  # max. allowed CTLE peaking (dB) (when optimizing, only)
MAX_CTLE_FREQ = 20.0  # max. allowed CTLE peak frequency (GHz) (when optimizing, only)

# - Simulation Control ----------------------------------------------------------
BIT_RATE = 10  # (Gbps)
NUM_BITS = 8000  # number of bits to run
PATTERN_LEN = 127  # repeating bit pattern length
SAMPLES_PER_BIT = 32  # samples per bit
NUM_AVG = 1  # Number of bit error samples to average, when sweeping.

MIN_BATHTUB_VAL = 1.0e-18
HPF_CORNER_COUPLING = (
    1.0e6
)  # Corner frequency of high-pass filter used to model capacitive coupling of periodic noise.

# - Tx ----------------------------------------------------------
OUTPUT_DRIVE_STRENGTH = 1.0  # output drive strength (Vp)
OUTPUT_IMPEDANCE = 100  # differential source impedance (Ohms)
OUTPUT_CAPACITANCE = (
    0.50
)  # parasitic output capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
PN_MAG = 0.001  # magnitude of periodic noise (V)
PN_FREQ = 0.437  # frequency of periodic noise (MHz)

# - Rx ----------------------------------------------------------
INPUT_IMPEDANCE = 100  # differential input resistance
INPUT_CAPACITANCE = (
    0.50
)  # parasitic input capacitance (pF) (Assumed to exist at both 'P' and 'N' nodes.)
AC_CAPACITANCE = (
    1.0
)  # a.c. coupling capacitance (uF) (Assumed to exist at both 'P' and 'N' nodes.)
BANDWIDTH = 12.0  # Rx signal path bandwidth, assuming no CTLE action. (GHz)
USE_DFE = True  # Include DFE when running simulation.
DFE_IDEAL = True  # DFE ideal summing node selector
PEAK_FREQ = 5.0  # CTLE peaking frequency (GHz)
PEAK_MAG = 10.0  # CTLE peaking magnitude (dB)
CTLE_OFFSET = 0.0  # CTLE d.c. offset (dB)

# - DFE ----------------------------------------------------------
DECISION_SCALER = 0.5
NUM_TAPS = 5  # Number of DFE taps
SUM_NUM_TAPS = 3  # Number of taps used in summing node filter.
GAIN = 0.5
DFE_NUM_AVG = 100
DFE_BW = 12.0  # DFE summing node bandwidth (GHz)

# - CDR ----------------------------------------------------------
DELTA_T = 0.1  # (ps)
ALPHA = 0.01
NUM_LOCK_AVG = 500  # number of UI used to average CDR locked status.
REL_LOCK_TOL = 0.1  # relative lock tolerance of CDR.
LOCK_SUSTAIN = 500

# - Analysis ----------------------------------------------------------
THRESHOLD = 6  # threshold for identifying periodic jitter spectral elements (sigma)
