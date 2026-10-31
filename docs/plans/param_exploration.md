# Exploration: parameters of the momentum (incl. frac10) and taker-flow books

Written on 2026-10-31. **Exploratory: decides nothing.** The registered papers and their decision
rules are unchanged. The screening rule was fixed in `scripts/run_param_exploration.py` (commit
42f3909) before running.

Run: `scripts/run_param_exploration.py` → `artifacts/research/param_exploration_v1/`. Shared engine,
0.06% fee, tiered slippage 3/6/12 bp, actual funding; development 2021-03-01 (taker 2021-12-01) to
2023-12-31, out of sample 2024-01-01 to 2026-10-19. 412 configurations, each also with costs ×2
(824 runs):
- momentum: lookback 14/21/30/45/60/90 × skip 0/1/3 × 10/15/20/30% per side × weekly/fortnightly ×
  equal/inverse-volatility weights, plus bands 0–0.10 at the primary;
- taker flow: window 3/5/7/10/14 days × 10/15/20/30% per side × rebalance 3/7/14 days × band
  0.02/0.05.

**Substantial improvement** = Sharpe +0.2 over the current configuration in both periods, worst
out-of-sample drawdown no more than 5 points worse, the median of its one-step grid neighbours +0.1
in both periods, and still better than the current with costs ×2.

## Result: no substantial improvement for any strategy

| Strategy | Current: Sharpe dev / OOS | Configurations that beat it in both periods | Meeting the rule |
|---|---|---|---|
| XS momentum | 0.33 / 1.00 | 4% (12 of 288) | 0 |
| XS frac10 | 0.44 / 1.32 | 1% (3 of 288) | 0 |
| Taker flow | 1.28 / 0.94 | 8% (9 of 120) | 0 |

- **The closest candidate** for momentum is skip 0 instead of 1 (30-day lookback, 15% per side):
  Sharpe 0.75 / 1.63, and it holds with costs ×2 (0.52 / 1.45). It fails the neighbour test: with a
  21- or 45-day lookback the same change gives development Sharpes of 0.24 and 0.00, and at 45–60
  days skip 1 is better. It is a peak, not a plateau.
- **Every momentum variant looks good out of sample and poor in development.** Out-of-sample Sharpe
  is 1.0–1.9 for most lookbacks, while the development Sharpe ranges from −0.65 to 0.75. The
  2024–2026 market favoured momentum in general; the differences between neighbouring parameters
  are mostly noise.
- **Band for momentum** (primary configuration): 0 → 0.21 / 0.88, 0.02 → 0.33 / 1.00, 0.05 → 0.57 /
  0.95, 0.10 → 0.41 / 1.23. A small, inconsistent effect; costs barely change (6.4–7.1%/yr), because
  momentum's turnover comes from changes in the ranking, not from resizing.
- **Taker flow:** the best combinations are a 10–14-day window with 15–30% per side, but none
  passes. They either worsen the drawdown or fail with costs ×2, and their neighbours average no
  better than the current configuration.

**Conclusion:** the current configurations are already at or near the top of their neighbourhoods,
and nothing found is robust enough to justify a new registered variant. Unlike TSMOM's band, where
a plausible mechanism (lower costs) held across a range of values, these differences do not
survive neighbouring parameters.
