# Registered hypothesis: open-interest signals (accumulated leverage)

*English translation. The Spanish original in [`es/`](es/open_interest_hypothesis.md) is the
dated record of the pre-registered rules.*

Written on 2026-10-21, **before downloading the open-interest data or looking at any result**.
Only the primary signal decides; the secondary ones are exploratory. The configuration and the
criteria are not changed after seeing the results.

## Data

Binance USDT-M public archives: `data.binance.vision/data/futures/um/daily/metrics/`. One file per
symbol and day, with rows every 5 minutes:
- `sum_open_interest`: open contracts, in units of the asset;
- `sum_open_interest_value`: open interest in USDT;
- `count_long_short_ratio`: long accounts / short accounts;
- `sum_toptrader_long_short_ratio`: the same, for top traders only;
- `sum_taker_long_short_vol_ratio`: aggressive buy volume / aggressive sell volume.

**Daily value for day *d*:** the last row of the UTC day stamped ≤ 23:55. It is known before the
signal close (24:00) and the fill (01:00).

**Universe:** only symbols in the monthly top 60 (`artifacts/research/universe_top60_v1`) are
downloaded, from one month before they enter. A symbol with no data that day is not eligible.

**Coverage:**
- The fraction of the universe with data is reported each month.
- The development period starts in the first month with coverage ≥ 80%, but never before
  2021-03-01.

## Primary signal: reversal of accumulated leverage (cross-sectional)

**Idea:**
- Strong growth in open contracts indicates new leveraged positions building up. Those positions
  tend to unwind (liquidations and forced closes), pushing price against the accumulated
  direction.
- Measuring in contracts rather than USDT separates new leverage from simple price moves.

**Score:** score_i(d) = ln(OI_i(d) / OI_i(d−7)), with OI = `sum_open_interest` (contracts).

**Portfolio:**
- **Pre-registered direction:** contrarian. **Short the 20% with the highest** contract growth
  and **buy the 20% with the lowest** growth (deleveraging).
- Construction identical to the primary momentum:
  - equal weights, 0.5 long / 0.5 short;
  - weekly rebalance with a 0.02 band;
  - at least 8 assets with a score.
- Tiered costs (3/6/12 bp + 0.06% fee) and actual funding.
- Decision at 23:00 UTC on day *d* and fill at 01:00 on day *d*+1, with the same
  `hf_research.engine`.

**Periods:** development from the start defined above to 2023-12-31; out of sample from
2024-01-01.

### Success criteria (all required)

| # | Criterion |
|---|---|
| 1 | Out-of-sample Sharpe ≥ 0.7 |
| 2 | Development Sharpe ≥ 0.4 |
| 3 | Out-of-sample worst drawdown ≥ −35% |
| 4 | Out-of-sample Sharpe without the 5 top-contributing symbols ≥ 0.4 |
| 5 | 20-day block bootstrap: probability of out-of-sample Sharpe ≤ 0 below 10% |

**Combination with the primary momentum:**
- Equal risk, with the scale computed on development data only.
- Useful if, out of sample, Sharpe ≥ momentum + 0.15 **and** the worst drawdown is better.
- The daily correlation with momentum is reported.

If the contrarian direction fails and the opposite direction (following OI growth) "works", that
does **not** count as a finding: it needs a new pre-registration and new data.

## Secondary signals (exploratory, decide nothing)

Built the same way as the primary and reported with the caveat that they are 3 extra tests: one
may come out well by chance.

| Signal | Score | Direction |
|---|---|---|
| S1. Retail positioning | `count_long_short_ratio` on day *d* | short the 20% most long, buy the 20% most short |
| S2. Aggressive flow | 7-day mean of ln(`sum_taker_long_short_vol_ratio`) | buy the 20% most buying, short the 20% most selling |
| S3. OI confirming price | primary score × sign of 7-day return | short where leverage grows with the move |

A secondary with out-of-sample Sharpe ≥ 0.7 only justifies a new pre-registration, which would
have to be validated in paper.

## Primary sensitivity (robustness only)

- Window: 3, 7 and 14 days.
- Fraction per side: 10%, 20% and 30%.
- Rebalance: weekly and every 3 days.
- Costs ×2.

The fraction of the 18 combinations with out-of-sample Sharpe ≥ 0.5 is reported.

## Known limitations

- **Source:** open interest is from Binance and execution would be on Bitget. Bitget positioning
  may differ.
- **Data quality:** Binance publishes these files with occasional gaps. A gap leaves the symbol
  out that day.
- **Inherited from momentum:** the universe and construction come from momentum, so results may
  correlate with it. That is why correlation and the combination are reported.

## Result (2026-10-21): primary rejected; one secondary (aggressive flow) merits a new pre-registration

Run: `scripts/run_open_interest.py` → `artifacts/research/open_interest_v1/`. Data:
`scripts/fetch_binance_metrics.py`, 144,638 symbol-days, 12.7% missing.

**Coverage:**
- Before 2021-12 the archive only has data for ~2% of the universe.
- Under the 80% rule, development starts on **2021-12-01**: 2 years.

### Primary: reversal of accumulated leverage (rejected on every criterion)

| Criterion | Threshold | Result | Met? |
|---|---|---|---|
| Out-of-sample Sharpe | ≥ 0.7 | −0.76 | no |
| Development Sharpe | ≥ 0.4 | −0.67 | no |
| Out-of-sample worst drawdown | ≥ −35% | −75% | no |
| Sharpe without the 5 best symbols | ≥ 0.4 | −1.43 | no |
| Bootstrap P(Sharpe ≤ 0) | < 10% | 86% | no |

- **No signal in either direction:**
  - The opposite direction, reported only as a control, also loses: −0.81 in development and
    −0.22 out of sample.
  - With turnover of 88 times capital per year, costs (~13%/yr) eat everything.
- **Grid:** 0 of 18 combinations with Sharpe ≥ 0.5, median −0.88. With rebalancing every 3 days
  and 10% per side, the account is wiped out.
- **Combination with momentum:** worse. Sharpe 0.03 vs 1.00.

### Secondaries (exploratory: 3 extra tests, decide nothing)

| Signal | Development Sharpe | Out-of-sample Sharpe | Worst drawdown (OOS) | Price / funding / costs OOS (%/yr) | Correlation with momentum (OOS) |
|---|---|---|---|---|---|
| S1 retail positioning (contrarian) | 1.87 | 0.73 | −39% | +8.7 / +16.8 / −5.4 | 0.38 |
| S2 aggressive flow (following) | 1.28 | 0.93 | −25% | +25.4 / +8.9 / −9.8 | 0.38 |
| S3 OI × price sign | −0.51 | −1.18 | −76% | −18.9 / −3.2 / −11.9 | — |

- **S2** (buy the coins with the most aggressive buying pressure over 7 days and short those with
  the most selling pressure):
  - The only one with a consistent price return: +27%/yr in development and +25%/yr out of
    sample.
  - Sharpe by year: 2022 1.51 · 2023 1.13 · 2024 −0.17 · 2025 0.66 · 2026 2.00.
  - Combined with momentum at equal volatility (exploratory): Sharpe 0.87 vs 0.22 in
    development, and 1.19 vs 1.03 out of sample.
- **S1:** out of sample it lives almost entirely off funding (+16.8%/yr). In practice it is a
  disguised carry. Also, 2024 had a Sharpe of −1.39.
- **S3:** as bad as the primary.

**Conclusion:**
- Leverage reversal does not exist in this data.
- S2 meets this pre-registration's rule (out-of-sample Sharpe ≥ 0.7), but it came out of 4 tests
  and the whole available history has already been seen.
- The only honest validation left is **a paper test with a new pre-registration**, because there
  is no unseen historical data.
