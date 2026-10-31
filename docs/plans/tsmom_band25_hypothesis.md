# Registered hypothesis: TSMOM BTC/ETH with a 0.25 band, validated on new data only

Written on 2026-10-31, before its paper's first rebalance. Like `xs_frac10_hypothesis.md`, this
variant **has already been seen over the whole history**: it came from the guardrail exploration
(`tsmom_guardrails_exploration.md`), which looked at bands from 0 to 0.40 on all available data. The
backtest cannot validate it. Validation is **exclusively the paper test**, on data after this
date, with the configuration frozen here.

**It changes nothing in the primary TSMOM paper** (`tsmom_paper_plan.md`) or its 2027-04-19
decision.

## Where it comes from

In the daily backtest engine (`hf_research.engine`, 0.06% fee, 3 bp slippage, actual funding):

| | Development Sharpe (2020-04 to 2023) | Out-of-sample Sharpe (2024+) | Worst drawdown dev / OOS | Turnover / cost per year (OOS) |
|---|---|---|---|---|
| Primary (band 0.05) | 1.09 | 0.63 | −26% / −34% | 36.6× / 3.3% |
| **Variant (band 0.25)** | **1.11** | **0.86** | **−20% / −25%** | 24.4× / 2.2% |

- With a wide band, the book trades mainly when the signal changes and mostly ignores the daily
  resizing to volatility, so it pays a third less in costs.
- The improvement is mostly **less risk, not more return**: daily correlation with the primary is
  0.93, and over the whole period the variant earns 0.9%/yr less.
- The result is noisy across neighbouring bands (out-of-sample Sharpe 0.58 at 0.15, 0.73–0.80 at
  0.20–0.25, 0.30 at 0.40), and the value was chosen after looking.

## Configuration (frozen)

Identical to the primary TSMOM except for the band:
- `TsmomConfig(band=0.25)`: lookbacks 20/60/120 days, 20% volatility target per asset, weight cap
  1.0, long/short.
- **Schedule and execution:** identical to the TSMOM paper (`scripts/paper_tsmom.py`). Daily
  decision on the 23:00 UTC Binance close, fill on d+1 at 01:00 UTC, simulated against the Bitget
  order book, 0.06% fee, actual Bitget funding. A fill more than 12 h late is missed.
- **Simulated capital:** 10,000 USDT. Book `tsmom_band25_v1`.

## Implementation gate (after 70 daily rebalances, ~2027-01-18)

Same thresholds as the TSMOM paper:
- ≥ 70 rebalances and ≤ 3 missed;
- median slippage ≤ 3 bp;
- daily correlation paper vs model ≥ 0.9;
- implementation gap ≥ −5%/yr.

## Decision (only with data from the variant's first rebalance)

| Date | Rule |
|---|---|
| 2027-05-03 (26 weeks) | **Discarded** if its worst drawdown is < −35% or its annualised Sharpe is < −0.5. Otherwise it keeps running. |
| 2027-11-01 (52 weeks) | **Replaces the primary TSMOM** in production (if the primary is live or approved) only if it meets everything: Sharpe ≥ the primary's + 0.2 on the same days, worst drawdown ≥ −30%, and implementation gate passed. |
| 2027-11-01 (52 weeks) | If the primary TSMOM was **discarded** on 2027-04-19, the variant can go to production on its own only if its paper alone has Sharpe ≥ 0.7 and worst drawdown ≥ −30% (the primary's original criteria) and the gate passed. |
| Otherwise | Discarded; the primary TSMOM's own plan is unaffected. |

**Power caveat:** the historical daily difference between the two has a Sharpe near zero (−0.08).
One year of paper will almost certainly not tell them apart on return, and the +0.2 Sharpe rule
will rarely be met. That is deliberate: the variant was found by looking at the data, so the bar to
replace a pre-registered configuration is high. Its realistic path to production is the second one
(the primary discarded, the variant passing the original criteria on fresh data).
