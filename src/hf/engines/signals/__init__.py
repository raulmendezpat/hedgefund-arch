from hf.engines.signals.flat import FlatSignalEngine
from hf.engines.signals.btc_trend_signal import BtcTrendSignalEngine
from hf.engines.signals.sol_bbrsi_signal import SolBbrsiSignalEngine

__all__ = [
    "FlatSignalEngine",
    "BtcTrendSignalEngine",
    "SolBbrsiSignalEngine",
    "PortfolioSignalEngine",
    "RegistryPortfolioSignalEngine",
]


def __getattr__(name: str):
    if name in {"PortfolioSignalEngine", "RegistryPortfolioSignalEngine"}:
        from hf.engines.signals.portfolio_signal import (
            PortfolioSignalEngine,
            RegistryPortfolioSignalEngine,
        )
        return {
            "PortfolioSignalEngine": PortfolioSignalEngine,
            "RegistryPortfolioSignalEngine": RegistryPortfolioSignalEngine,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
