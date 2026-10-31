# Exploration: how the guardrails change the BTC/ETH TSMOM book

Written on 2026-10-31. **Exploratory: decides nothing.** The TSMOM paper test, its configuration
and its decision rule on 2027-04-19 (`tsmom_paper_plan.md`) stay as registered. Lowering the
success thresholds after seeing the results would be overfitting, so the thresholds themselves are
not studied here; this measures what the risk guardrails and sizing limits do to performance.

Run: `scripts/run_tsmom_guardrails.py` → `artifacts/research/tsmom_guardrails_v1/`. An hour-by-hour
account with the live runner's rules (`hf_research.live_xs.check_risk`), primary `TsmomConfig()`
targets, 0.06% fee + 3 bp slippage and actual funding. Development 2020-04-01 to 2023-12-31; out of
sample 2024-01-01 to 2026-10-19, each from equity 1. Without guardrails it reproduces the backtest
(Sharpe 1.12 / 0.59, against 1.10 / 0.63 in `tsmom_btc_eth_hypothesis.md`).

## Findings

| Variant | Dev Sharpe | OOS Sharpe | OOS CAGR | Worst hourly DD dev / OOS | Kills |
|---|---|---|---|---|---|
| No guardrails | 1.12 | 0.59 | 13.7% | −30% / −36% | 0 |
| **Current** (kill −35%, 7-day pause, daily −8%) | 1.12 | 0.59 | 13.9% | −30% / −35% | 1 |
| Kill −15%, 7-day pause | 1.01 | 0.55 | 12.4% | −15% / −16% | 13 |
| Kill −20%, 7-day pause | 1.09 | 0.61 | 14.5% | −20% / −21% | 6 |
| Kill −25%, 7-day pause | 1.11 | 0.60 | 14.2% | −25% / −25% | 3 |
| Kill −25%, 30-day pause | 1.12 | 0.45 | 9.2% | −25% / −25% | 3 |
| Kill −35%, 30-day pause | 1.12 | 0.42 | 8.3% | −30% / −35% | 1 |
| Kill −35%, stop for good | 1.12 | −0.13 | −3.7% | −30% / −35% | 1 |
| Daily loss 3–12% (live rule) | 1.12 | 0.59 | 13.9% | −30% / −35% | 1 |
| Max weight 0.25 | 1.15 | 0.47 | 7.8% | −25% / −27% | 0 |
| Max weight 0.5 | 1.13 | 0.59 | 13.5% | −29% / −35% | 1 |
| Band 0.0 | 1.10 | 0.54 | 12.3% | −31% / −35% | 1 |
| Band 0.2 | 1.30 | 0.73 | 18.1% | −22% / −32% | 0 |

1. **The kill switch level barely matters; what happens after it does.** With a 7-day review
   pause, any level from −20% to −35% gives the same Sharpe. A tighter level caps the drawdown
   almost exactly (−20% → −21%), at a small cost in return. A 30-day pause misses the rebound and
   costs 0.15–0.17 of Sharpe. Stopping for good after the first hit turns the out-of-sample result
   negative. The production plan's rule (a pause for review, not a retirement) is the right one.
2. **The daily-loss guard never acts.** The live runner measures the day's loss from its first run
   of the UTC day (00:17) and the TSMOM fill is at 01:00, so an 8%, 5% or even 3% limit changes
   nothing. A rolling-24h version changes almost nothing either. It protects against an intraday
   crash between runs, not against daily losses.
3. **The weight cap only matters well below 1.0.** Weights rarely exceed 0.75. At 0.25 the drawdown
   falls to −27%, but out-of-sample return halves.
4. **A wider band is the one change that improves both periods**, but noisily
   (`band_sweep.txt`):

   | Band | 0.05 (current) | 0.10 | 0.15 | 0.175 | 0.20 | 0.225 | 0.25 | 0.30 | 0.40 |
   |---|---|---|---|---|---|---|---|---|---|
   | Dev Sharpe | 1.12 | 1.19 | 1.05 | 1.15 | 1.30 | 1.17 | 1.24 | 1.08 | 1.04 |
   | OOS Sharpe | 0.59 | 0.55 | 0.58 | 0.61 | 0.73 | 0.80 | 0.78 | 0.70 | 0.30 |

   Between 0.2 and 0.3 the out-of-sample Sharpe is 0.70–0.80, with a smaller drawdown (−19% to
   −32%). Mechanism: with a wide band the book mostly trades when the signal changes and ignores
   the volatility resizing, so it pays fewer costs. But with two assets and a handful of large
   trades, results jump between neighbouring values (0.15 is worse than 0.05 in development), and
   the value was found by looking at the whole history.

## What to do with it

- Nothing changes in the registered TSMOM paper or its 2027-04-19 decision.
- The wide band is a lead for a **new** pre-registered variant (for example band 0.25), validated
  only on paper data from its registration date, like `xs_frac10_hypothesis.md`.
- The daily-loss rule should be described as an intraday crash guard, or measured on a rolling
  window, if it is meant to limit daily losses.
