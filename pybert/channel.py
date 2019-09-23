"""A serDes channel consists of a driver, a channel and a receiver."""
from logging import getLogger

import numpy as np
from numpy import array, diff, exp, pad, real, where, zeros
from numpy.fft import fft, ifft
from pybert.materials import TwistedCopperPair24Gauge
from pybert.equalization import get_tap_fir_numerator
from pybert.utility import calc_G, calc_gamma, import_channel, make_ctle, trim_impulse
from traits.api import Array, Bool, File, Float, HasTraits, Int, cached_property


class Channel(HasTraits):
    """docstring for Channel"""

    def __init__(self):
        super(Channel, self).__init__()
        self.log = getLogger("pybert.buffer")
        self.log.debug("Creating Channel")
        self.use_ch_file = Bool(False)  #: Import channel description from file? (Default = False)
        self.padded = Bool(False)  #: Zero pad imported Touchstone data? (Default = False)
        self.windowed = Bool(False)  #: Apply windowing to the Touchstone data? (Default = False)
        self.f_step = Float(
            10
        )  #: Frequency step to use when constructing H(f). (Default = 10 MHz)
        self.ch_file = File(
            "", entries=5, filter=["*.s4p", "*.S4P", "*.csv", "*.CSV", "*.txt", "*.TXT", "*.*"]
        )  #: Channel file name.
        self.impulse_length = Float(
            0.0
        )  #: Impulse response length. (Determined automatically, when 0.)
        material = TwistedCopperPair24Gauge()  # TODO: Make this a menu option to swap materials.
        self.Rdc = Float(material.dc_resistance_per_meter)  #: Channel d.c. resistance (Ohms/m).
        self.w0 = Float(material.w_transition_freq)  #: Channel transition frequency (rads./s).
        self.R0 = Float(material.skin_effect_resistance)  #: Channel skin effect resistance (Ohms/m).
        self.Theta0 = Float(material.loss_tangent)  #: Channel loss tangent (unitless).
        self.Z0 = Float(
            material.characteristic_impedance
        )  #: Channel characteristic impedance, in LC region (Ohms).
        self.v0 = Float(material.rel_velocity)  #: Channel relative propagation velocity (c).
        self.l_ch = Float(material.channel_length)  #: Channel length (m).

        self.len_h = Int(0)
        self.chnl_dly = Float(0.0)  #: Estimated channel delay (s).
        self.chnl_h = Array()
        self.chnl_H = Array()
        self.chnl_trimmed_H = Array()
        self.start_ix = Int(0)

    @cached_property
    def _get_tx_h_tune(self):
        nspui = self.nspui

        taps = get_tap_fir_numerator(self.eq.tx_tap_tuners)

        h = sum([[x] + list(zeros(nspui - 1)) for x in taps], [])

        return h

    # This function has been pulled outside of the standard Traits/UI "depends_on / @cached_property" mechanism,
    # in order to more tightly control when it executes. I wasn't able to get truly lazy evaluation, and
    # this was causing noticeable GUI slowdown.
    def calc_chnl_h(self):
        """
        Calculates the channel impulse response.

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

        t = self.t
        ts = t[1]
        nspui = self.nspui
        impulse_length = self.impulse_length * 1.0e-9

        if self.use_ch_file:
            chnl_h = import_channel(self.ch_file, ts, self.padded, self.windowed)
            if chnl_h[-1] > (max(chnl_h) / 2.0):  # step response?
                chnl_h = diff(chnl_h)  # impulse response is derivative of step response.
            chnl_h /= sum(chnl_h)  # Normalize d.c. to one.
            chnl_dly = t[where(chnl_h == max(chnl_h))[0][0]]
            chnl_h.resize(len(t))
            chnl_H = fft(chnl_h)
        else:
            l_ch = self.l_ch
            v0 = self.v0 * 3.0e8
            R0 = self.R0
            w0 = self.w0
            Rdc = self.Rdc
            Z0 = self.Z0
            Theta0 = self.Theta0
            w = self.w
            Rs = self.tx.output_impedance
            Cs = self.tx.output_capacitance * 1.0e-12
            RL = self.rx.input_impedance
            Cp = self.rx.input_capacitance * 1.0e-12
            CL = self.rx.cac * 1.0e-6

            chnl_dly = l_ch / v0
            gamma, Zc = calc_gamma(R0, w0, Rdc, Z0, v0, Theta0, w)
            H = exp(-l_ch * gamma)
            chnl_H = 2.0 * calc_G(
                H, Rs, Cs, Zc, RL, Cp, CL, w
            )  # Compensating for nominal /2 divider action.
            chnl_h = real(ifft(chnl_H))

        min_len = 10 * nspui
        max_len = 100 * nspui
        if impulse_length:
            min_len = max_len = impulse_length / ts
        chnl_h, start_ix = trim_impulse(chnl_h, min_len=min_len, max_len=max_len)
        chnl_h /= sum(chnl_h)  # a temporary crutch.
        temp = chnl_h.copy()
        temp.resize(len(t))
        chnl_trimmed_H = fft(temp)

        chnl_s = chnl_h.cumsum()
        chnl_p = chnl_s - pad(chnl_s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0))

        self.chnl_h = chnl_h
        self.len_h = len(chnl_h)
        self.chnl_dly = chnl_dly
        self.chnl_H = chnl_H
        self.chnl_trimmed_H = chnl_trimmed_H
        self.start_ix = start_ix
        self.t_ns_chnl = array(t[start_ix : start_ix + len(chnl_h)]) * 1.0e9
        self.chnl_s = chnl_s
        self.chnl_p = chnl_p

        return chnl_h

    @cached_property
    def _get_ctle_h_tune(self):
        w = self.w
        len_h = self.len_h
        rx_bw = self.eq.rx_bw_tune * 1.0e9
        peak_freq = self.eq.peak_freq_tune * 1.0e9
        peak_mag = self.eq.peak_mag_tune
        offset = self.eq.ctle_offset_tune
        mode = self.eq.ctle_mode_tune

        _, H = make_ctle(rx_bw, peak_freq, peak_mag, w, mode, offset)
        h = real(ifft(H))[:len_h]
        h *= abs(H[0]) / sum(h)

        return h

    @cached_property
    def _get_ctle_out_h_tune(self):
        chnl_h = self.chnl_h
        tx_h = self.eq.tx_h_tune
        ctle_h = self.eq.ctle_h_tune

        tx_out_h = np.convolve(tx_h, chnl_h)
        h = np.convolve(ctle_h, tx_out_h)

        return h
