Modules in *pybert* package
---------------------------

pybert - Main *PyBERT* class definition, as well as some helper classes.
************************************************************************

.. automodule:: pybert.pybert

.. autoclass:: pybert.pybert.StoppableThread
   :members:

.. autoclass:: pybert.pybert.TxOptThread
   :members:

.. autoclass:: pybert.pybert.RxOptThread
   :members:

.. autoclass:: pybert.pybert.CoOptThread
   :members:

.. autoclass:: pybert.pybert.TxTapTuner
   :members:

.. autoclass:: pybert.pybert.PyBERT
   :members:

controller - Model control logic.
***********************************

.. automodule:: pybert.controller
   :members: my_run_sweeps, my_run_simulation, update_results, update_eyes

view - Main GUI window layout definition.
************************************************

.. automodule:: pybert.view
   :members: MyHandler, RunSimThread

utility - Various utilities used by other modules.
******************************************************

.. automodule:: pybert.utility
   :members: moving_average, find_crossing_times, find_crossings, calc_jitter, make_uniform, calc_gamma, calc_G, calc_eye, make_ctle, trim_impulse, import_channel, interp_time, import_time, sdd_21, import_freq, lfsr_bits, safe_log10, pulse_center

plots - Plot definitions for the *PyBERT* GUI.
****************************************************

.. automodule:: pybert.plots

help - Contents of the *Help* tab of the *PyBERT* GUI.
*************************************************************

.. automodule:: pybert.help

configuration - Data structure for saving *PyBERT* configuration.
**************************************************************

.. automodule:: pybert.configuration
   :members:

results - Data structure for saving *PyBERT* results.
*********************************************************

.. automodule:: pybert.results
   :members:

dfe - DFE behavioral model.
***************************

.. automodule:: pybert.dfe
   :members: LfilterSS, DFE

cdr - CDR behavioral model.
***************************

.. automodule:: pybert.cdr
   :members: CDR
