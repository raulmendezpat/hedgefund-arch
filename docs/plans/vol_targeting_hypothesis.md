# Registered hypothesis: volatility targeting of momentum and x1–x4 leverage

*English translation. The Spanish original in [`es/`](es/vol_targeting_hypothesis.md) is the
dated record of the pre-registered rules.*

Written on 2026-10-21, **before building the backtest or looking at any result** of this
strategy. The primary configuration, the success criteria and the rules for accepting a leverage
level are not changed after seeing the results. The grid measures robustness; it is not used to
choose parameters.

## Idea

Cross-sectional momentum has sharp, concentrated drawdowns: the worst ones arrive when the market
is most volatile. Volatility targeting shrinks exposure when the strategy's recent volatility rises
and grows it when volatility falls, to keep risk roughly constant (Barroso & Santa-Clara, 2015,
"Momentum has its moments"). If it works, it improves the Sharpe and reduces drawdowns without
changing the signal.

## Primary configuration

**Base strategy:** the primary momentum (`PRIMARY_STRATEGY`), unchanged:
- top-60 universe;
- weekly rebalance;
- tiered costs of 3/6/12 bp plus a 0.06% fee;
- 0.5 long / 0.5 short (gross exposure 1.0).

**Scale:** set at each rebalance, when deciding on day *d*. The base's target weights are
multiplied by

  k_d = min(cap, target / σ̂_d)

where:
- **σ̂_d** is the annualised volatility (daily deviation × √365) of the unscaled base strategy's
  net daily returns over the previous **60 days**. It uses only rows ≤ *d*−1, whose last fill is
  before the decision. With fewer than 30 days, k = 1.
- **target** = **25%** annual. This is roughly the base's development volatility (27.7%), so
  average exposure stays near 1.
- **cap** = **2.0**: maximum gross exposure is 2.

**Periods:** development 2021-03-01 to 2023-12-31; out of sample from 2024-01-01.

## Volatility-targeting success criteria (x1, all required)

| # | Criterion |
|---|---|
| 1 | Out-of-sample Sharpe ≥ base + 0.10 |
| 2 | Development Sharpe ≥ base (no worse) |
| 3 | Out-of-sample worst drawdown, **at the same realised volatility as the base**, better than the base's |
| 4 | Out-of-sample worst month, at the same volatility, better than the base's |

For criterion 3, the targeted series is statically rescaled to the base's realised volatility over
the same period. That way it is not rewarded simply for taking less risk.

Also reported, not a criterion: a 20-day block bootstrap of the probability that the
out-of-sample Sharpe improvement is ≤ 0.

## Leverage x1, x2, x3, x4

Applied to the base and to the targeted version (target × L, cap × L). Weights are multiplied by L
and go through the same engine: fee, slippage and funding scale with exposure.

**Hourly account simulation:**
- Each day takes the weights held since the last fill and each symbol's hourly closes.
- From those, equity is rebuilt hour by hour: the intraday minimum, not just the daily close.

**Liquidation:**
- Cross margin. Liquidation occurs if at any hour equity ≤ maintenance margin × current gross
  notional.
- Maintenance margin: **2%** of notional. This is conservative for alts on Bitget, whose first
  tier is between 0.5% and 1.5%.
- A liquidation ends the simulation for that period.

**Production plan kill switch:** the first hour at which drawdown from peak reaches −25% is
recorded.

**Metrics per period:**
- CAGR, Sharpe, worst daily and hourly drawdown, and worst day.
- Liquidation date, if any, and date of the first kill-switch hit.
- Distribution of 30-day rolling-window returns: median, 5th and 95th percentile, share of
  windows ≥ +10% and ≤ −10%.

**A leverage level is considered acceptable for production** only if it meets all three
conditions, **in development and out of sample**:
1. no liquidation;
2. worst hourly drawdown > −40%;
3. worst 30-day window > −30%.

## Sensitivity (robustness only)

- σ̂ window: 20, 60 and 120 days.
- Cap: 1.5, 2 and 3.
- Target fixed at 25%.

The fraction of the 9 combinations whose out-of-sample Sharpe beats the base is reported.

## Known limitations

- Volatility is estimated on the unscaled base strategy (a shadow). Its returns differ slightly
  from the scaled ones because of the band and costs.
- Hourly candles are from Binance; execution would be on Bitget.
- Margin tiers by position size and exchange ADL are not modelled.
- Slippage does not grow with size. At x3 and x4, a large account would pay more impact than
  modelled.

## Result (2026-10-21): volatility targeting rejected; leverage > x1 not acceptable

Run: `scripts/run_vol_targeting.py` → `artifacts/research/vol_targeting_v1/`.

### Volatility targeting (x1)

| Criterion | Base | Targeted | Met? |
|---|---|---|---|
| Out-of-sample Sharpe (≥ base + 0.10) | 1.00 | 0.71 | no |
| Development Sharpe (≥ base) | 0.33 | 0.48 | yes |
| Out-of-sample worst drawdown, same volatility | −32.4% | −36.5% | no |
| Out-of-sample worst month, same volatility | −11.9% | −14.5% | no |

- **Bootstrap:** probability that the out-of-sample Sharpe improvement is ≤ 0 = 98%. The median
  improvement is −0.28.
- **Grid:** none of the 9 combinations beats the base out of sample; the median is 0.71.
- **Reading:** momentum in this universe lacks the "high volatility → bad return" structure that
  targeting exploits. Its best weeks (positive tails) come with high volatility, and targeting
  trims them. This is consistent with the "lottery" character already observed: return depends on
  a few large up-moves.

### Leverage (account simulated hour by hour, 2% maintenance, each period starting at 1)

| Base | Development CAGR | Out-of-sample CAGR | Worst hourly drawdown (dev / OOS) | Worst 30-day window | 30-day windows ≥ +10% / ≤ −10% (OOS) |
|---|---|---|---|---|---|
| x1 | 5.5% | 31% | −26% / −35% | −19% | 19% / 4% |
| x2 | 4.5% | 51% | −46% / −59% | −37% | 32% / 17% |
| x3 | 0.2% | 63% | −62% / −76% | −54% | 37% / 26% |
| x4 | −11% | 63% | −76% / −86% | −70% | 41% / 32% |

- **Liquidations:** none. Losses arrive gradually, not all at once.
- **Bigger drawdowns, no extra return:** drawdowns grow proportionally, but return does not keep
  up. In development, x3 and x4 end at zero or at a loss because of volatility drag.
- **Verdict:** only x1, for both the base and the targeted version, meets the pre-registered
  rules. x2 to x4 are rejected.

### Finding for the production plan

At x1, the −25% kill switch would have fired in both periods: 2021-08-26 in development and
2024-08-28 out of sample (worst hourly drawdown −26% and −35%). With that threshold, the strategy
would have stopped before its best stretch (2025–2026). This is left as an open decision in
`xs_momentum_production_plan.md` §4.
