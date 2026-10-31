# Registered hypothesis: cross-sectional momentum at 10% per side, validated on new data only

*English translation. The Spanish original in [`es/`](es/xs_frac10_hypothesis.md) is the dated
record of the pre-registered rules.*

Written on 2026-10-24, before its paper's first rebalance (2026-10-26). Like taker flow, this
variant **has already been seen over the whole history**: it came from a diagnostic of momentum's
selection run on the same backtest data. The backtest cannot validate it. Validation is
**exclusively the paper test**, on data after this date and with the configuration frozen here.

## Where it comes from

Diagnostic of the primary hypothesis's selection (`xs_momentum_production_plan.md`), week by week
since 2021-03, with each token's 7-day P&L net of funding and relative to the universe average:
- The edge is concentrated at the extremes of the ranking: the top 5% and the bottom 5%. The 5–20%
  band on each side contributes about zero.
- What sits just outside the cut (percentiles 20–30 and 70–80) performs like the middle of the
  ranking: the cut does not leave good signals out.
- Liquidity ranks 61–120 perform worse than the top 60: widening the universe does not help.

Backtest with the shared engine and backtest costs (0.06% fee, 3 bp slippage), top 60:

| | Development Sharpe (2021-03 to 2023) | Out-of-sample Sharpe (2024+) | Worst drawdown dev / OOS | Annual cost |
|---|---|---|---|---|
| Primary (20% per side) | 0.42 | 1.07 | −24% / −32% | 4.0% |
| **Variant (10% per side)** | **0.51** | **1.38** | **−39% / −32%** | 4.7% |

- By year, the variant wins in 2021, 2024 and 2026 and loses in 2022, 2023 and 2025. The
  improvement comes from a few good years.
- Daily correlation with the primary: 0.82–0.87.
- The −39% development drawdown exceeds the production kill switch (−35%).

**What is given up:** with ~55–60 tradable symbols there are ~6 per side and a weight of ~8.3% per
position, double the primary's. Per-token tail risk (see §5 of the production plan) weighs twice
as much.

## Configuration (frozen)

Identical to the primary hypothesis except for the fraction per side:
- **Signal:** 30-day return with a 1-day skip, on Binance 1h closes at 23:00 UTC.
- **Universe:** monthly Binance USDT-M top 60 by volume, restricted to symbols listed on Bitget.
- **Portfolio:** long the **10%** with the highest score and short the **10%** with the lowest,
  at least 2 per side; equal weights 0.5 / −0.5; at least 8 assets with a score; 0.02 band.
- **Schedule and execution:** identical to the momentum paper. Decision on Sunday, fill on Monday
  at 01:00 UTC, simulated against the Bitget order book, 0.06% fee and actual Bitget funding.
- **Simulated capital:** 10,000 USDT.

## Implementation gate (2027-01-11, 12 rebalances)

Same thresholds as the momentum paper:
- ≥ 12 rebalances and ≤ 1 missed;
- median slippage per tier ≤ 3 / 6 / 12 bp;
- daily correlation paper vs model ≥ 0.9;
- implementation gap ≥ −5%/yr.

## Decision (only with data from the first rebalance, 2026-10-26)

It is compared against the primary's paper (`xs_momentum_v1`) on the same days, because the
question is whether the variant is **better than the primary**, not just whether it makes money.

| Date | Rule |
|---|---|
| 2027-04-19 (26 weeks) | **Discarded** if its worst drawdown is < −35%, or its annualised Sharpe is < −0.5. Otherwise it keeps running. |
| 2027-10-25 (52 weeks) | **Replaces the primary** in production only if it meets everything: Sharpe ≥ 0.7, Sharpe ≥ the primary's + 0.3 on the same days, worst drawdown ≥ −35% and implementation gate passed. Otherwise it is discarded and the primary stays as is. |

**Requirement before production:** the runner's sanity limit aborts on weights > 10%. With fewer
than 50 tradable symbols there are 5 per side (exactly 10%), and with fewer than 45, 4 per side
(12.5%), which would abort the rebalance. That limit must be reviewed before going live.

**Power caveat:** the daily difference between the two has a historical Sharpe of 0.4–1.1. Over 52
weeks that gives a t-statistic of 0.4–1.1: even if the improvement were real, one year of paper
probably cannot tell it from noise. The rule deliberately demands a large improvement, because the
variant doubles concentration and a false positive costs real capital.

## Variant discarded before the paper: longs only in tokens listed for less than a year

It came from the same diagnostic. Longs in tokens listed less than 365 days ago (by Binance's
`onboardDate`) outperformed older ones: +79 bp/week vs +1 in development and +401 vs −65 out of
sample. As a portfolio (shorts as in the primary, longs only in young tokens from the top 20%), it
was **worse than the primary in development**: Sharpe 0.31 vs 0.49, with a −42% vs −18% drawdown,
before costs. The reason is that there are very few: a median of 3 longs per week, and fewer than
2 in 54 of 289 weeks.

Decision of 2026-10-24: **not registered**. It is noted here so it is not retested without a new
reason.
