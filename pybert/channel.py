"""A serDes channel consists of a driver, a channel and a receiver."""
from functools import lru_cache
from logging import getLogger

import numpy as np
from numpy.fft import fft, ifft

from pybert.materials import MATERIALS
from pybert.utility import calc_G, calc_gamma, import_channel, trim_impulse


class Channel:
    """docstring for Channel"""

    def __init__(self):
        super(Channel, self).__init__()
        self.log = getLogger("pybert.channel")
        self.log.debug("Initializing Channel")
        self.use_ch_file: bool = False  #: Import channel description from file? (Default = False)
        self.padded: bool = False  #: Zero pad imported Touchstone data? (Default = False)
        self.windowed: bool = False  #: Apply windowing to the Touchstone data? (Default = False)
        self.f_step = 10.0  #: Frequency step to use when constructing H(f). (Default = 10 MHz)
        self.ch_file = (
            None
        )  #: Channel file name. "*.s4p", "*.S4P", "*.csv", "*.CSV", "*.txt", "*.TXT", "*.*"
        self.impulse_length = 0.0  #: Impulse response length. (Determined automatically, when 0.)
        self.material = MATERIALS["UTP_24Gauge"]

    @lru_cache(maxsize=None)
    def calc_chnl_h(self, t, nspui, w, tx, rx):
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

        ts = t[1]
        impulse_length = self.impulse_length * 1.0e-9

        if self.use_ch_file:
            chnl_h = import_channel(self.ch_file, ts, self.padded, self.windowed)
            if chnl_h[-1] > (max(chnl_h) / 2.0):  # step response?
                chnl_h = np.diff(chnl_h)  # impulse response is derivative of step response.
            chnl_h /= sum(chnl_h)  # Normalize d.c. to one.
            chnl_dly = t[np.where(chnl_h == max(chnl_h))[0][0]]
            chnl_h.resize(len(t))
            chnl_H = fft(chnl_h)
        else:
            l_ch = self.material.channel_length
            rel_velocity = self.material.rel_velocity * 3.0e8
            skin_effect_resistance = self.material.skin_effect_resistance
            w_transition_freq = self.material.w_transition_freq
            dc_resistance_per_meter = self.material.dc_resistance_per_meter
            characteristic_impedance = self.material.characteristic_impedance
            loss_tangent = self.material.loss_tangent
            output_impedance = tx.output_impedance
            output_capacitance = tx.output_capacitance * 1.0e-12
            input_impedance = rx.input_impedance
            input_capacitance = rx.input_capacitance * 1.0e-12
            CL = rx.cac * 1.0e-6

            chnl_dly = l_ch / rel_velocity
            gamma, Zc = calc_gamma(
                skin_effect_resistance,
                w_transition_freq,
                dc_resistance_per_meter,
                characteristic_impedance,
                rel_velocity,
                loss_tangent,
                w,
            )
            H = np.exp(-l_ch * gamma)
            chnl_H = 2.0 * calc_G(
                H,
                output_impedance,
                output_capacitance,
                Zc,
                input_impedance,
                input_capacitance,
                CL,
                w,
            )  # Compensating for nominal /2 divider action.
            chnl_h = np.real(ifft(chnl_H))

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
        chnl_p = chnl_s - np.pad(chnl_s[:-nspui], (nspui, 0), "constant", constant_values=(0, 0))

        chnl_h = chnl_h
        len_h = len(chnl_h)
        chnl_trimmed_H = chnl_trimmed_H
        t_ns_chnl = np.array(t[start_ix : start_ix + len(chnl_h)]) * 1.0e9

        return chnl_dly, start_ix, t_ns_chnl, chnl_H, chnl_s, chnl_p, len_h
