# Exploration: model and pipeline changes to raise return without much more risk

Written on 2026-11-02. Exploratory: it decides nothing for the registered papers, and nothing here
changes a runner. Run: `scripts/run_pipeline_ideas.py` → `artifacts/research/pipeline_ideas_v1/`.
The screening rule was committed before running (5cbed55). An idea passes if, in development AND
out of sample:
- CAGR improves by ≥ 2 points/yr;
- Sharpe does not fall;
- the worst drawdown is at most 3 points worse.

For the funding filter, all three thresholds must pass and the gain must hold with costs ×2.

Periods:
- **XS and frac10:** development 2021-03 to 2023, out of sample 2024-01 to 2026-10-19.
- **Combinations:** from 2021-12, so that the taker book trades in all of them.

Costs are the engine's: 0.06% fee, tiered slippage (3/6/12 bp; 3 bp for TSMOM) and Binance
funding.

## A. Funding filter (XS, frac10): rejected

**The idea:** drop a selected long whose trailing 7-day funding is above +x/yr, or a selected
short whose funding is below −x/yr, and give its slot to the next-ranked asset. It was prompted by
the paper's LSK short, which paid ~9% of its notional in funding in one week.

| Book | x | CAGR dev / OOS | Sharpe dev / OOS | DD dev / OOS |
|---|---|---|---|---|
| XS | — (current) | +5.5% / +31.4% | 0.33 / 1.00 | −24% / −32% |
| XS | 50% | −3.1% / +27.3% | 0.02 / 0.93 | −30% / −32% |
| XS | 100% | +11.2% / +27.6% | 0.52 / 0.93 | −24% / −33% |
| XS | 200% | +5.4% / +28.6% | 0.33 / 0.94 | −25% / −33% |
| frac10 | — (current) | +10.2% / +71.6% | 0.44 / 1.32 | −41% / −34% |
| frac10 | 100% | +14.6% / +56.4% | 0.54 / 1.14 | −42% / −41% |

- **Out of sample, it hurts at every threshold.** Only 100% helps in development, and even there
  the neighbouring thresholds do not.
- The likely reason: extreme funding marks crowded positions, and those are often momentum's
  strongest moves. A short that pays heavy funding is usually one that keeps falling.

## B. Maker execution: large bound, worth measuring

An upper bound that assumes every order fills as maker. "Half maker" assumes half the volume does.
Unfilled orders and adverse selection are not modelled, so this is never registered from a
backtest.

| Book | CAGR dev / OOS: current | Maker bound (2 bp, no slippage) | Half maker (4 bp, half slippage) |
|---|---|---|---|
| XS | +5.5% / +31.4% | +11.5% / +39.0% | +8.5% / +35.2% |
| frac10 | +10.2% / +71.6% | +17.5% / +83.5% | +13.8% / +77.5% |
| Taker flow | +23.1% / +23.6% | +33.0% / +34.5% | +28.0% / +28.9% |
| TSMOM | +10.3% / +15.3% | +12.5% / +18.3% | +11.4% / +16.8% |

- **Costs are the largest controllable drag.** The weekly books turn over ~100× a year, so each
  basis point saved per trade is worth about 1 point/yr. Drawdowns improve slightly.
- **What is unknown is the fill rate.** A post-only order at the touch fills only if the price
  comes to it, and the orders that do not fill are the ones the market ran away from.
- **The way to measure it** is on the real account once it trades. Post a post-only order at the
  best bid or ask, cancel after N minutes, and send the remainder as taker. Log the fill rate and
  the cost against the mid at the decision time.
- That is a change to the live runner. It should not be made before the 2026-11-09 gate or during
  the first weeks live, and it needs its own registration.

## C. Combining books: more return at the same risk, but only with books that pass their papers

Equal capital, daily rebalanced. "At XS dev vol" scales the combination up to the XS book's
development volatility, which needs gross exposure above 1x.

| Book | CAGR dev / OOS | Sharpe dev / OOS | DD dev / OOS |
|---|---|---|---|
| XS alone (from 2021-12) | +1.1% / +31.4% | 0.16 / 1.00 | −24% / −32% |
| XS + TSMOM | +5.2% / +25.7% | 0.35 / 1.11 | −16% / −25% |
| XS + taker | +10.3% / +29.2% | 0.71 / 1.17 | −10% / −23% |
| XS + TSMOM + taker | +10.2% / +26.2% | 0.72 / 1.29 | −11% / −19% |
| XS + taker at XS vol (×1.47) | **+14.5% / +42.8%** | 0.71 / 1.17 | −15% / −32% |
| XS + TSMOM + taker at XS vol (×1.51) | **+14.8% / +40.0%** | 0.72 / 1.29 | −17% / −27% |

Daily correlations, out of sample:
- XS–taker 0.38;
- XS–TSMOM 0.12;
- TSMOM–taker −0.01.

- **Without scaling,** combining improves Sharpe and roughly halves the drawdowns, but out of
  sample it earns less than XS alone.
- **Scaled to the XS book's risk,** the three-book combination passes the rule: +9 to +14
  points/yr at an equal or better drawdown.
- **The condition is that taker flow and TSMOM pass their own papers.** Taker flow's development
  Sharpe (1.28) carries most of the gain. Taker flow, TSMOM and frac10 were all found or chosen
  on history, so none of their gains count until their papers confirm them. If they fail, this
  falls with them.
- **Scaling to ×1.5 means gross exposure above 1x.** That needs:
  - the hourly-account check of `leverage_1x_10x.md` on the combined book (liquidation, −40%
    hourly drawdown, −30% worst 30 days);
  - changing the live sanity limit (gross ≤ 1x).

  This is not proposed before the original gate.

## Conclusion

- **The funding filter is discarded.**
- **The two real levers are execution costs and diversification:**
  - Maker execution is measured live, with its own registration, after the first weeks live.
  - The combination is decided when taker flow and TSMOM reach their paper decisions (2027).
- **Nothing changes in the current papers or in the 2026-11-09 decision.**
