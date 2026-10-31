# Registered hypothesis: time-series momentum (TSMOM) on BTC and ETH

*English translation. The Spanish original in [`es/`](es/tsmom_btc_eth_hypothesis.md) is the
dated record of the pre-registered rules.*

Written on 2026-10-21, **before building the backtest or looking at any result** of this
strategy. The primary configuration and the success criteria are not changed after seeing the
results. The grid measures robustness; it is not used to choose parameters.

## Idea

Each asset is bought when its own recent trend is up and shorted when it is down (Moskowitz, Ooi
& Pedersen, 2012). Unlike cross-sectional momentum, it does not compare assets with each other: it
bets that the BTC and ETH trends persist. It is interesting for two reasons:
- **Liquidity:** they are the two most liquid assets, with minimal costs and no delisting risk.
- **Diversification:** it should have little correlation with cross-sectional momentum, which is
  market-neutral, so it could serve as a second leg of the portfolio.

## Primary configuration

**Assets:** Binance USDT-M BTC/USDT and ETH/USDT perpetuals. Hourly candles and funding are
complete since listing (BTC 2019-09, ETH 2019-11).

**Signal** (at the 23:00 UTC close of day *d*, using only closes ≤ *d*):
- s_i = mean of sign(r_20), sign(r_60) and sign(r_120), where r_L is the asset's L-day return.
- s ∈ {−1, −⅓, +⅓, +1}.

**Size:** w_i = s_i × 0.20 / σ_i, capped at |w_i| ≤ 1.0.
- σ_i is the annualised volatility of the asset's daily returns over the last 60 days.
- Each asset targets ~20% volatility and the portfolio 25–30%, similar to cross-sectional
  momentum.

**Execution:**
- Daily decision and fill at 01:00 UTC; the same `hf_research.engine` as momentum.
- 0.05 band: a same-side position is not resized if the change is smaller.
- Long/short mode.

**Costs:**
- 0.06% fee plus 3 bp slippage per side. This is tier 1, since BTC and ETH are the most liquid.
- Actual funding for each asset.

**Periods:**
- Development: 2020-04-01 to 2023-12-31. It starts there because ETH needs 120 days of history.
- Out of sample: from 2024-01-01.

## Success criteria (all required)

| # | Criterion |
|---|---|
| 1 | Out-of-sample Sharpe ≥ 0.7 |
| 2 | Out-of-sample worst drawdown ≥ −30% |
| 3 | Development Sharpe ≥ 0.5 |
| 4 | Each asset on its own (same rule, BTC only or ETH only) with out-of-sample Sharpe > 0 |

**Combination with the primary cross-sectional momentum:**
- Equal risk: TSMOM is scaled to momentum's development volatility.
- Useful if, out of sample, the combined Sharpe ≥ momentum alone + 0.15 **and** the worst
  drawdown is better.
- The implied gross exposure is reported. If matching the risk needs gross exposure above 2, it is
  declared not implementable.

**Also reported, not a criterion:**
- Sharpe and worst drawdown of a 50/50 BTC/ETH buy-and-hold at the same volatility, and the
  correlation with it.
- 20-day block bootstrap: probability that the out-of-sample Sharpe is ≤ 0.
- Hourly account simulation (`hf_research.vol_target`): worst hourly drawdown and whether it hits
  the −35% kill switch.
- Sharpe by year and by side: contribution of longs and of shorts.

## Sensitivity (robustness only)

- Windows: 20 only, 60 only, 120 only, and the 20/60/120 combination.
- Band: 0.02, 0.05 and 0.10.
- Mode: long/short and long only.
- Double costs on the primary configuration.

The fraction of the 24 combinations with out-of-sample Sharpe ≥ 0.5 is reported.

## Known limitations

- **Few independent bets:** only two assets, highly correlated (~0.8). The number of bets is low
  and confidence intervals are wide. A 2–3 year result may depend on one or two trends.
- **Regime:** out of sample (2024–2026) is mostly a bull market. That is why the contribution of
  longs and shorts is split out and compared with buy-and-hold.
- **Data source:** data is from Binance and execution would be on Bitget. For BTC and ETH the price
  and funding differences are small.

## Result (2026-10-21): narrowly rejected; the most robust candidate so far

Run: `scripts/run_tsmom.py` → `artifacts/research/tsmom_btc_eth_v1/`.

| Criterion | Threshold | Result | Met? |
|---|---|---|---|
| Out-of-sample Sharpe | ≥ 0.7 | 0.63 | no |
| Out-of-sample worst drawdown | ≥ −30% | −33.7% | no |
| Development Sharpe | ≥ 0.5 | 1.10 | yes |
| Each asset out of sample > 0 | > 0 | BTC 0.46 · ETH 0.70 | yes |

| Combination with cross-sectional momentum (out of sample) | Momentum alone | Combined | Met? |
|---|---|---|---|
| Sharpe (≥ +0.15) | 1.00 | 1.11 (+0.11) | no |
| Worst drawdown | −32.4% | −24.6% | yes |
| TSMOM gross exposure at scale (p95 ≤ 2) | — | 0.93 | yes |

In development the combination also improves: Sharpe 0.55 vs 0.33 and worst drawdown −16% vs
−24%. The out-of-sample daily correlation with momentum is 0.12.

**Why it is the most robust of everything tested** (even without passing):
- **Years:** positive Sharpe in all 7, from 2020 to 2026: 2.14 · 1.03 · 0.41 · 0.66 · 0.43 · 0.58 ·
  1.01.
- **Grid:**
  - All 24 combinations have a positive out-of-sample Sharpe; 62% exceed 0.5.
  - Medians: 0.57 out of sample and 1.19 in development.
- **Costs ×2:** out-of-sample Sharpe 0.52. Turnover is low and BTC/ETH costs are minimal.
- **Bootstrap:** probability that the out-of-sample Sharpe is ≤ 0 = 13%.
- **Not dependent on one asset:** BTC and ETH are positive separately.
- **Not disguised beta:** out-of-sample correlation with buy-and-hold is −0.06. Buy-and-hold had a
  Sharpe of 0.39, with a −38% drawdown at the same volatility.
- **Hourly account:** worst drawdown −34% out of sample, missing the −35% kill switch by very
  little.

**Weaknesses:**
- **Return by side:**
  - Development: longs +35%/yr, shorts +0.5%/yr.
  - Out of sample: longs +17%/yr, shorts +6.5%/yr.
  - Shorts only contributed in the recent period.
- **Funding:** costs 2.4–6.5%/yr, because longs pay funding.
- **Positive months:** 38% out of sample. It wins with a few long trends.

**Post-hoc readings (do not justify changing the primary):** the long-only variant (0.85) and the
60-day-only variant (0.92) had better out-of-sample Sharpe. Choosing them now would be overfitting
to a mostly bullish period.

**Conclusion:** the hypothesis, as registered, does not pass. To continue, the honest route is to
validate it on new data: a paper test of the unchanged primary configuration, with its own
pre-registered gate. Adjusting the thresholds is not appropriate.
