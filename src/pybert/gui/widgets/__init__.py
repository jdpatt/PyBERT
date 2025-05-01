"""PyBERT GUI widgets package."""

from .analysis import AnalysisConfigWidget
from .channel import ChannelConfigWidget
from .debug_console import DebugConsoleWidget
from .rx import RxConfigWidget
from .rx_optimization_ctle import RxOptimizationCTLEWidget
from .rx_optimization_dfe import RxOptimizationDFEWidget
from .simulation import SimulationControlWidget
from .tx import TxConfigWidget
from .tx_optimization import TxOptimizationWidget

__all__ = [
    "SimulationControlWidget",
    "TxConfigWidget",
    "ChannelConfigWidget",
    "RxConfigWidget",
    "AnalysisConfigWidget",
    "TxOptimizationWidget",
    "RxOptimizationCTLEWidget",
    "RxOptimizationDFEWidget",
    "DebugConsoleWidget",
]
