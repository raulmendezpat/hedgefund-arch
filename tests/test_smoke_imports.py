def test_imports():
    import hf
    from hf.core.types import Signal, RegimeState, Allocation, Candle
    from hf.core.interfaces import SignalEngine, RegimeEngine, CapitalAllocator, PortfolioEngine
    assert hf.__version__
