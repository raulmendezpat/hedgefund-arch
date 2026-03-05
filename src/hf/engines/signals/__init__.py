"""Signal engines (entry/exit logic).

Paquete dedicado a generación de señales.
Step-1 scaffold: engine plano (flat) para no cambiar el comportamiento actual.
"""

from .flat import FlatSignalEngine
from .btc_trend_signal import BtcTrendSignalEngine
from .sol_bbrsi_signal import SolBbrsiSignalEngine

__all__ = ["FlatSignalEngine", "BtcTrendSignalEngine", "SolBbrsiSignalEngine"]
