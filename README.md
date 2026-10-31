# hedgefund-arch

*English summary below · [Versión en español](#español)*

## English

A systematic trading research and execution platform, built and run end to end by one person in Python.

**Core architecture** (`src/hf`, `src/hf_core`): signal engines (bots), a market regime engine, a capital allocator and a portfolio engine, with machine-learning scoring (scikit-learn gradient boosting and logistic regression, validated walk-forward with an embargo; `src/hf_score`, `src/hf_core/ml`).

**Research**
- A backtest engine (`src/hf_research/engine.py`) charges exchange fees, tiered slippage and historical funding on every change in position. It also handles assets that get delisted.
- Each hypothesis is written down before it is tested. The plans in `docs/plans/` set the test windows, the pass/fail criteria and the decision dates in advance.
- Strategies studied include cross-sectional momentum, time-series momentum, funding carry, open interest, taker flow, volatility targeting, and trend and stock momentum on Bitget's traditional-asset perpetuals (stocks, ETFs, metals, energy, FX). Rejected hypotheses are kept on record together with the reason they failed.

**Validation**
- Paper-trading runners (`scripts/paper_*.py`) track each candidate strategy in real time, on live market data and with simulated fills. They check whether the backtest's cost assumptions hold in live markets.

**Execution**
- A live runner (`scripts/live_xs_momentum.py`, `src/hf_research/live_xs.py`) defaults to dry-run mode and only places orders when two explicit switches are both set.
- Its risk controls are:
  - a kill switch that triggers on drawdown;
  - a reduce-only mode after a large daily loss;
  - sanity limits that abort a rebalance;
  - deposits and withdrawals are not counted as profit or loss;
  - it refuses to trade when it finds positions it did not open.

**Operations**
- Scheduled jobs (`deploy/`) run every runner from a pinned git worktree.
- An hourly health check (`scripts/healthcheck_xs.py`) alerts on missed runs, overdue rebalances, errors in the logs, outsized position losses and low disk space.
- A weekly review (`scripts/event_review_xs.py`) uses the Claude API to check news and contract mappings for each open position. It only raises alerts and never places trades.

**History**
- Work started in February 2026 in [LiveTradingBots](https://github.com/raulmendezpat/LiveTradingBots), building on the open-source bot template by Robot Traders (its 2024–2025 commits are theirs). It was carried into this repo in March 2026.
- The first version, an ML-scored multi-bot portfolio, ran live on AWS EC2 until August 2026.
- It was shut down because it was losing money, and the instance was deleted.
- The current strategies replaced it. Each was pre-registered with fixed pass/fail criteria before testing, then validated in paper trading and a live dry-run before any capital goes back in.

**Documentation:** the research and production plans are in [`docs/plans/`](docs/plans/), translated from the dated Spanish originals in [`docs/plans/es/`](docs/plans/es/). Start with [`xs_momentum_production_plan.md`](docs/plans/xs_momentum_production_plan.md).

**Tests:** `python -m pytest tests/`, run on every push by GitHub Actions (`.github/workflows/tests.yml`).

## Español

Plataforma de investigación y ejecución de trading sistemático:

- **Investigación:** el motor de backtest descuenta comisiones, slippage y funding. Cada hipótesis se registra antes de probarla. Los planes originales en español están en `docs/plans/es/`; `docs/plans/` tiene la traducción al inglés.
- **Validación:** ejecutores en paper (`scripts/paper_*.py`).
- **Ejecución:** ejecutor en vivo con dry-run por defecto y controles de riesgo (`scripts/live_xs_momentum.py`).
- **Historia:** el trabajo empezó en febrero de 2026 en LiveTradingBots, a partir de la plantilla abierta de Robot Traders. La primera versión operó en vivo en AWS EC2 hasta agosto de 2026. Se apagó porque daba pérdida y se reemplazó por estrategias registradas antes de probarlas y validadas en paper.
- **Operación:** agentes programados (`deploy/`), healthcheck horario y revisión semanal de eventos con la API de Claude.
