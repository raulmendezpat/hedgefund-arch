#!/usr/bin/env python3
from hf.engines.legacy_wrappers import LegacyBacktestPortfolioEngine

def main() -> None:
    LegacyBacktestPortfolioEngine().run_legacy_backtest()

if __name__ == "__main__":
    main()
