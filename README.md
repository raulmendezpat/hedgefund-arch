# hedgefund-arch

*English summary below · [Versión en español](#español)*

## English

A systematic trading research and execution platform, built and run end to end by one person in Python.

**Research**
- A backtest engine (`src/hf_research/engine.py`) charges exchange fees, tiered slippage and historical funding on every change in position. It also handles assets that get delisted.
- Each hypothesis is written down before it is tested. The plans in `docs/plans/` set the test windows, the pass/fail criteria and the decision dates in advance.
- Strategies studied include cross-sectional momentum, time-series momentum, funding carry, open interest, taker flow and volatility targeting. Rejected hypotheses are kept on record together with the reason they failed.

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

**Documentation:** the research and production plans are in [`docs/plans/`](docs/plans/), translated from the dated Spanish originals in [`docs/plans/es/`](docs/plans/es/). Start with [`xs_momentum_production_plan.md`](docs/plans/xs_momentum_production_plan.md).

**Tests:** `python -m pytest tests/`

## Español

Plataforma de investigación y ejecución de trading sistemático:

- **Investigación:** el motor de backtest descuenta comisiones, slippage y funding. Cada hipótesis se registra antes de probarla. Los planes originales en español están en `docs/plans/es/`; `docs/plans/` tiene la traducción al inglés.
- **Validación:** ejecutores en paper (`scripts/paper_*.py`).
- **Ejecución:** ejecutor en vivo con dry-run por defecto y controles de riesgo (`scripts/live_xs_momentum.py`).
- **Operación:** agentes programados (`deploy/`), healthcheck horario y revisión semanal de eventos con la API de Claude.
