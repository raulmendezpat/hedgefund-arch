# Registered hypothesis: delta-neutral funding carry

*English translation. The Spanish original in [`es/`](es/funding_carry_hypothesis.md) is the
dated record of the pre-registered rules.*

Written on 2026-10-20, **before building the backtest or looking at any result** of this
strategy. The primary configuration and the success criteria are not changed after seeing
results. The sensitivity grid measures robustness; it is not used to choose parameters.

## Idea

When a perpetual's funding is persistently positive, leveraged longs pay the shorts. The strategy
is **short the perpetual and long the same amount in spot**: it has no net price exposure and
collects the funding. There are three risks:
- funding falls or reverses while the position is open;
- the basis (perpetual minus spot) moves against the position;
- the costs of entering and exiting.

## Primary configuration

**Universe:**
- Each month, the Binance USDT-M top 60 by volume, the same as in momentum
  (`artifacts/research/universe_top60_v1`).
- Symbols without a `<BASE>/USDT` spot pair priced on Binance that day are excluded.

**Signal** (at the 23:00 UTC close of day *d*, without look-ahead):
- Sum of realised funding over the last 7 days, counting only events dated UTC ≤ *d*.
- Annualised as the 7-day sum × 365/7.

**Entry and exit** (with hysteresis):
- Enter if annualised funding is ≥ **10%**.
- Exit if it falls below **3%** or the symbol leaves the universe.
- At most **10 positions**. If there are more candidates, the highest-funding ones are chosen.
  An open position is only replaced when it leaves under the exit rule.

**Size:**
- Equal weights: **0.07 of capital per pair** (spot long 0.07 + perpetual short 0.07).
- With 10 pairs, 0.70 of capital is in spot and the rest is perpetual margin (3x leverage on the
  short, roughly 0.23).
- With fewer than 10 candidates, the unused part stays in cash, earning 0.

**Execution:**
- Daily decision at 23:00 UTC and execution at 01:00 UTC the next day, as in momentum.
- 0.02 band: weights are not readjusted for small differences.

**Costs:**
- Perpetual: 0.06% taker fee.
- Spot: 0.10% taker fee.
- Plus slippage by liquidity tier on each leg: 3 / 6 / 12 bp for universe ranks 1–10 / 11–30 /
  31–60.

**Periods:**
- Development: 2021-03-01 to 2023-12-31.
- Out of sample: 2024-01-01 to the latest data.

## Success criteria (out of sample, after costs)

**All** must be met:

| Criterion | Threshold |
|---|---|
| Sharpe | ≥ 1.5 |
| Positive months | ≥ 70% |
| Worst drawdown | no worse than −10% |
| Sharpe without the 5 top-contributing symbols | ≥ 1.0 |
| Development Sharpe (consistency) | ≥ 0.5 |

**Reported, but not a criterion:**
- Annual return. Something on the order of 5–25% is expected; it is not optimised.
- Turnover, annual cost, average number of positions and months with no position.
- **Margin risk:** the worst perpetual loss between two executions, as a fraction of available
  margin, to see whether a price spike could liquidate the short before rebalancing.

## Combination test with momentum

The carry's net daily series is combined with the primary momentum's, with **equal risk
weights**: each strategy scaled to the same volatility, computed on development data only.

The combination is considered useful if it meets **both** conditions out of sample:
- Combined Sharpe ≥ momentum-only Sharpe + 0.2.
- Worst drawdown better than momentum alone.

## Sensitivity (robustness only)

| Parameter | Values |
|---|---|
| Entry threshold | 5%, 10%, 20% |
| Window | 3, 7, 14 days |
| Maximum positions | 5, 10, 20 |
| Costs | double the primary |

The fraction of the grid with out-of-sample Sharpe above 1.0 is reported.

## Known limitations

- **Binance data:** the backtest uses Binance spot and funding; execution would be on Bitget.
  Bitget funding was ~40% higher with a 0.4 correlation to Binance's, so the backtest would need
  repeating on Bitget data before a paper test.
- **Margin in two accounts:** spot and perpetual live in separate accounts unless the unified
  account is used. A price spike creates a loss on the short and a gain on the spot in different
  accounts. That is why margin risk is reported.
- **Cash earns nothing:** stablecoin yield on idle cash is not modelled.

## Result (2026-10-20): hypothesis rejected

Run: `scripts/run_funding_carry.py` → `artifacts/research/funding_carry_v1/`. 337 top-60
perpetuals; 270 with a Binance spot pair; 46 delisted contracts.

**Data correction (not a parameter change).** After a delisting, Binance keeps serving perpetual
candles frozen at the settlement price, with zero volume. The first run kept those pairs open with
a broken hedge; DEGO alone cost −8.9%. The perpetual is now treated as dead from the start of its
final run of flat, zero-volume candles (≥24 h), and the pair is closed there. Before the
correction the out-of-sample Sharpe was −0.37.

| Criterion (out of sample) | Threshold | Result | Met? |
|---|---|---|---|
| Sharpe | ≥ 1.5 | 1.13 | no |
| Positive months | ≥ 70% | 38% | no |
| Worst drawdown | ≥ −10% | −3.7% | yes |
| Sharpe without the 5 best symbols | ≥ 1.0 | 1.14 | yes |
| Development Sharpe | ≥ 0.5 | 3.73 | yes |

- **Return:** out-of-sample CAGR 2.7% with 2.3% volatility. Funding +5.8%/yr, costs −3.4%/yr,
  price +0.2%/yr: the hedge works.
- **The edge is dying out:**
  - Sharpe by year: 2021 8.6 · 2022 −2.6 · 2023 2.3 · 2024 7.8 · 2025 0.6 · 2026 −1.9.
  - Since 2025 almost every month is slightly negative: average funding no longer pays for the
    turnover.
- **Few positive months despite a positive Sharpe:** return is concentrated in a few high-funding
  months (2024-03 +3.4%, 2025-03 +3.3%). The rest are small cost-driven losses.
- **Costs ×2:** out-of-sample Sharpe 0.34.
- **Grid:**
  - 59% of configurations with out-of-sample Sharpe ≥ 1; median 1.33.
  - Those with a 20% entry earn more, between 8 and 13%/yr.
  - This is a post-hoc reading, so it does not justify changing the primary.
- **Combination with momentum:**
  - On paper it passes: out-of-sample Sharpe 1.49 vs 1.00, and worst drawdown −24% vs −32%.
  - But it requires scaling the carry ×19 to match momentum's volatility, about 13 times capital
    in notional per leg.
  - With a short that rose 93% in one day (1000SHIB) and a perpetual loss of up to 44% of free
    margin even unlevered, that scale is not implementable. **It is not considered valid.**
- **Delisting risk:** even with the correction, DEGO cost −1.5%. Spot collapsed while the
  perpetual was still in the universe. It is a real risk of the strategy, not an artefact.

**Conclusion:** delta-neutral funding carry, with realistic costs, earns 2–4%/yr with very low
volatility, and the 2025–2026 trend is negative. It does not go to paper.
