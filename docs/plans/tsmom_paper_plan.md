# Registered plan: TSMOM BTC/ETH paper test and its combination with cross-sectional momentum

*English translation. The Spanish original in [`es/`](es/tsmom_paper_plan.md) is the dated record
of the pre-registered rules.*

Written on 2026-10-21, before the paper test started. Nothing below changes while it runs.

## Why paper-test a hypothesis that did not pass

TSMOM did not meet its pre-registered criteria (`tsmom_btc_eth_hypothesis.md`): out-of-sample
Sharpe 0.63 vs 0.7, and drawdown −33.7% vs −30%. Even so, it is the most robust strategy tested:
positive every year, in all 24 variants and with costs ×2. The thresholds are not lowered.
Instead, **new data** is collected with the primary configuration intact, and **the same
criteria** are then reapplied over a longer out-of-sample period.

## What runs

**TSMOM book** (`runtime/paper/tsmom_btc_eth_v1`):
- Simulated capital: 10,000 USDT.
- Configuration: primary `TsmomConfig()`, unchanged.
  - 20/60/120-day signal on Binance 23:00 UTC closes.
  - 20% target volatility per asset, weight cap 1.0.
  - 0.05 band, long/short mode.
- Execution simulated against the Bitget order book (BTC/USDT and ETH/USDT), with a 0.06% fee.
  Actual Bitget funding is applied.
- Daily rebalance: decision on day *d*, fill on day *d*+1 at 01:00 UTC. A fill more than 12 h
  late is recorded as missed.
- It never sends orders: it uses public data only.

**Combination:**
- Evaluated from the two papers, without a third book: daily return = 0.5 × cross-sectional
  momentum (`xs_momentum_v1`) + 0.5 × TSMOM.
- The TSMOM scale is 1.0: in the backtest it came out at 0.98 to match development volatility.
- Compared with cross-sectional momentum alone, on the same days.

## Implementation gate (2027-01-11, together with cross-sectional momentum's)

| Check | Threshold |
|---|---|
| Daily rebalances done | ≥ 70 |
| Rebalances missed | ≤ 3 |
| Median slippage (BTC/ETH, tier 1) | ≤ 3 bp |
| Daily correlation paper vs model (replay on Binance data) | ≥ 0.9 |
| Implementation gap | ≥ −5%/yr |

Twelve weeks are not enough to judge profitability. This gate only validates that the paper
reproduces the model.

## Decision on the combination (2027-04-19, after 26 weeks)

1. Re-run `scripts/run_tsmom.py` with no changes to code or criteria. The out-of-sample period is
   extended from 2024-01-01 to the cut-off date.
2. The combination goes to production (with the capital and scaling of the cross-sectional
   momentum plan) only if both conditions hold:
   - the TSMOM implementation gate has passed;
   - with the extended period, **either** the primary TSMOM meets its 4 criteria **or** the
     combination meets its own: Sharpe ≥ momentum + 0.15, better drawdown and implementable
     exposure.
3. Otherwise TSMOM is discarded and only cross-sectional momentum remains.

**Caveat:** re-evaluating the same criteria with more data is a sequential test. It adds a second
chance to pass. This is accepted because the criteria and the code do not change, and because half
of the new data (the paper's) is genuinely out of sample.
