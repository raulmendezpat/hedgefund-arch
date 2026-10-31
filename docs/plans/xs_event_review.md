# Weekly event and mapping review of the momentum book (alerts only)

*English translation. The Spanish original is in [`es/`](es/xs_event_review.md).*

Written on 2026-10-28, before its first run. It is a monitor: **it does not change the registered
strategy or the orders**. It runs separately from the runner, and if it fails, the rebalance goes
ahead as usual.

## What it does

Every Monday, between the close of the Sunday 23:00 UTC candle (00:00 UTC) and the 01:00 UTC
fill, it computes the week's target portfolio with the same code as the paper and the runner
(`xs_signal.compute_targets`) and checks two things:

1. **Binance ↔ Bitget mapping** (every tradable member of the universe, ~60):
   - compares the price on both exchanges;
   - the ratio must be 1, or the power of 10 implied by the prefix (`1000PEPE` on Binance against
     `PEPE` on Bitget gives 1/1000);
   - if it deviates by more than 2%, the two symbols are probably not the same asset (reused
     ticker, redenomination or wrong alias) and it alerts.
   - This is a fixed rule, with no LLM: more reliable than asking a model.
2. **Events** (each position in the portfolio, ~24):
   - Claude (`claude-haiku-5-5`, `low` effort, up to 2 web searches per token) looks for discrete
     events in the last 14 days or scheduled for the next 7: delistings and listings, hacks,
     unlocks, migrations or redenominations, regulatory actions, depegs, team sales, airdrops and
     upgrades;
   - it ignores price commentary and technical analysis;
   - it rates the risk **for the position's side** (a new listing is a risk for a short; an
     unlock, for a long) as `none`, `low`, `medium` or `high`, citing the source.

**Notifications:** a summary on every run, and an alert with sound for each doubtful mapping and
each position rated `medium` or `high`. If it runs after the fill (for example, because the Mac
was off), it says so.

## What to do with an alert

There is no mechanism to exclude a position, and none is added: that would change the strategy.
With a serious alert before the fill, the only available action is to pause the live runner that
week (`launchctl unload` of its agent), which counts as a missed rebalance. It is a manual decision
by the account owner.

## Cost

- **Until 2026-10-31** it used `claude-sonnet-5-5` with up to 3 searches: ~0.035–0.045 USD per
  position, about 1.04 USD per week (~54 USD a year). With 300 USDT live that is ~18% of capital
  a year, nearly all the expected return.
- **Since 2026-10-31** it uses `claude-haiku-5-5` with up to 2 searches. Measured on 6 positions:
  0.006–0.010 USD per position, about 0.15–0.25 USD per week (~8–13 USD a year): ~3–4% of 300 USDT
  and ~1% of 1,000.
- Both models' ratings vary between runs (the same token can come out `none`, `low` or `medium`),
  so little is lost by switching models.
- Each run stores its estimated cost in the report. It is an estimate: some searches run inside the
  code tool and the search counter may not reflect them. Actual spend is visible in the Anthropic
  console.

## How it is measured

Each review is stored in `runtime/event_review/reviews/<decision_day>.json`, with the portfolio,
the ratings and the sources. **On 2027-05-03 (26 weeks):** the paper's weekly P&L of flagged
positions (`medium`/`high`) is compared with the rest.
- It is descriptive: with few alerts per week there will be no statistical power.
- Using the alerts to trade (for example, excluding flagged positions) would be a new hypothesis,
  with its own registration and its own paper test.
