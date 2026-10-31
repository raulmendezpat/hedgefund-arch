# Registered test: leverage 1x to 10x for each strategy

Written on 2026-10-31, before running it. The four strategies' 1x results are already known (their
own hypothesis files), so this test does not look for an edge: it measures what each leverage level
does to the account path, and which levels are survivable. The acceptance rule is fixed here and is
the one already used in `vol_targeting_hypothesis.md`. It is not changed after seeing the results.

## Strategies (frozen configurations, unchanged)

| Strategy | Configuration | Development | Out of sample |
|---|---|---|---|
| XS momentum (primary) | `PRIMARY_STRATEGY`, top-60 universe, weekly, band 0.02 | 2021-03-01 to 2023-12-31 | 2024-01-01 to latest |
| XS frac10 | same with 10% per side | same | same |
| Taker flow | S2 of `open_interest_hypothesis.md`, weekly, band 0.02 | 2021-12-01 to 2023-12-31 (data coverage) | same |
| TSMOM BTC/ETH | primary `TsmomConfig()`, daily, band 0.05 | 2020-04-01 to 2023-12-31 | same |

Costs as in each backtest: 0.06% fee, tiered slippage (3/6/12 bp; 3 bp for TSMOM) and actual
Binance funding.

## Method

- **Leverage L = 1, 2, …, 10:** each strategy's target weights are multiplied by L, and so is its
  band. Fees, slippage and funding scale with exposure through the same engine.
- **Hourly account** (`hf_research.vol_target.account_path`), from the weights held each day and
  each symbol's hourly closes. Each period starts at equity 1.
  - Cross margin. Liquidation when, at any hour, equity ≤ maintenance × gross notional.
  - Maintenance 2% of notional (conservative; Bitget's first tier is 0.5–1.5% for most of the
    universe). Reported alongside, not deciding: 1%.
  - The −35% kill switch is not applied to the path: the first hour that hits it is reported.

**A leverage level is acceptable** for a strategy only if, **in development and out of sample**:
1. there is no liquidation;
2. the worst hourly drawdown is better than −40%;
3. the worst 30-day window is better than −30%.

**Also reported, not deciding:** CAGR, Sharpe, final equity, date the −35% kill switch would first
fire, share of 30-day windows ≥ +10% and ≤ −10%, and the full 2021–2026 path from equity 1 (from
each strategy's first development day).

## Known limitations

- Slippage does not grow with order size. At high leverage, a large account would pay more market
  impact than modelled.
- Bitget's margin tiers by position size and auto-deleveraging (ADL) are not modelled. Five current
  universe symbols allow at most 5–10x per position, which in cross margin limits the initial
  margin but not the total book.
- Hourly marks miss intra-hour wicks, which can trigger liquidations sooner than modelled at high
  leverage.
- To trade above 1x, the live runner's sanity limits (gross ≤ 1x, weight ≤ 10%) and the Bitget
  leverage setting (3x) would have to change.
