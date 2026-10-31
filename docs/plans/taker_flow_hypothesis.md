# Registered hypothesis: cross-sectional taker flow, validated on new data only

*English translation. The Spanish original in [`es/`](es/taker_flow_hypothesis.md) is the dated
record of the pre-registered rules.*

Written on 2026-10-21. Unlike the earlier hypotheses, this signal **has already been seen over the
whole available history**: it is secondary S2 of `open_interest_hypothesis.md`, which came out
with a Sharpe of 1.28 in development and 0.93 out of sample, chosen from 4 tests. The backtest
cannot validate it. Validation is **exclusively the paper test**, on data after this date and with
the configuration frozen here.

## Configuration (frozen; identical to the S2 backtest)

**Universe:** the monthly Binance USDT-M top 60 (same calculation as momentum), restricted to
symbols listed on Bitget.

**Data:**
- `buySellRatio` = aggressive buy volume / aggressive sell volume, in 5-minute bars.
- Source: the Binance API (`/futures/data/takerlongshortRatio`).
- It matches the backtest's `sum_taker_long_short_vol_ratio` archive at the same timestamp
  (maximum difference 0.0016, from rounding).

**Score** at the Sunday close (day *d*):
- For each UTC day, the mean of ln(ratio) over its 5-minute bars.
- Then the mean of those daily values over the 7 days *d*−6..*d*, with at least 5 days of data.

**Portfolio:**
- Buy the 20% with the highest score and short the 20% with the lowest.
- Equal weights, 0.5 long / 0.5 short.
- At least 8 assets with a score; a symbol without a score is not eligible.
- 0.02 band.

**Schedule and execution:** identical to the momentum paper.
- Decision on Sunday and fill on Monday at 01:00 UTC.
- Simulated against the Bitget order book, 0.06% fee and actual Bitget funding.
- Simulated capital: 10,000 USDT.

## Implementation gate (2027-01-11, 12 rebalances)

Same thresholds as the momentum paper:
- ≥ 12 rebalances and ≤ 1 missed;
- median slippage per tier ≤ 3 / 6 / 12 bp;
- daily correlation paper vs model ≥ 0.9;
- implementation gap ≥ −5%/yr.

The comparison model uses the weights the paper decided each week, with Binance fill prices and
the backtest's costs. That way it measures execution only.

## Decision on the signal (only with data from the first rebalance, 2026-10-26)

| Date | Rule |
|---|---|
| 2027-04-19 (26 weeks) | **Discarded** if the paper's annualised Sharpe is < −0.5. Otherwise it keeps running. |
| 2027-10-25 (52 weeks) | Goes to production, as a second leg next to momentum with half the risk each, only if the paper has **Sharpe ≥ 0.7** and **worst drawdown ≥ −30%**, and the implementation gate has passed. Otherwise it is discarded. |

**Also reported, without deciding anything:** the 50/50 combination with the momentum paper on the
same days, and the correlation between the two.

**Power caveat:** one year of weekly data is little. With a true Sharpe of 0.7, the probability of
observing ≥ 0.7 over 52 weeks is ~50%. The rule is deliberately demanding: a false positive costs
real capital, while a false negative only costs an opportunity.
