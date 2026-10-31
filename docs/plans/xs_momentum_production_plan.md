# Production plan: cross-sectional momentum on Bitget

*English translation. The Spanish original in [`es/`](es/xs_momentum_production_plan.md) is the
dated record of the pre-registered rules.*

Written on 2026-10-20, **before the paper book's first rebalance** (2026-10-26). The rules are set
now so that the results cannot influence them. Changing them later requires recording the reason
here, with a date, **before** looking at the data that would motivate the change.

Decisions marked *proposal, to be confirmed* are the defaults; they still need the account
owner's confirmation.

**Amendment of 2026-10-26 (§8):** accelerated start with 300 USDT on 2026-11-16, subject to a
short gate; no scaling until the original gate of 2027-01-11 is met.

## 0. What goes into production

The registered hypothesis, unchanged:
- **Signal and portfolio:** long/short on 30-day return with a 1-day skip; 20% of symbols per
  side, equal weights.
- **Schedule:** weekly rebalance, decided at the Sunday 23:00 UTC close and executed on Monday
  at 01:00 UTC.
- **Band:** 0.02.
- **Universe:** Binance top 60 by volume, listed on Bitget.
- **Exposure:** 1x gross, Bitget leverage 3x, cross margin.
- **Code:** `scripts/live_xs_momentum.py` in `live` mode, at a pinned commit.

**What is known about the strategy** (stress tests):
- Out-of-sample Sharpe of 1.08, falling to 0.86–1.01 with realistic costs.
- The 5–95% bootstrap interval runs from 0.28 to 1.99.
- 5 new tokens explained 79% of the out-of-sample return.

In short: there is an edge, but it is fragile and concentrated. The plan is designed for that:
small capital at the start, slow scaling and clear stop criteria.

## 1. Entry gate (end of the paper period)

Review after the paper's 12th rebalance (fill of **2027-01-11**). **All** items must be met:

| Criterion | Threshold | Source |
|---|---|---|
| Rebalances executed | ≥ 12 | `paper_xs_momentum.py report` |
| Rebalances missed | ≤ 1 | same |
| Median slippage per tier (top 10 / 11–30 / 31–60) | ≤ 3 / 6 / 12 bp | same |
| Daily correlation paper vs model | ≥ 0.9 | same |
| Implementation gap paper vs model | ≥ −5%/yr | same |
| Dry-run: rebalances with orders generated without error | ≥ 4 | `live_xs_momentum.py status` |
| Dry-run: unexplained aborts, sanity aborts or Python errors | 0 | monitor + logs |

**What it measures and what it does not:**
- 12 weeks do **not** allow judging profitability. With a Sharpe of 1, the standard error of a
  12-week return is larger than the expected return.
- The gate measures whether **the implementation reproduces the model**, not whether it makes
  money.
- The risk that the strategy does not work is controlled by capital size (§2–3) and by the stop
  criteria (§4).

**If a criterion fails:** no move to production. The cause is documented. If it is fixable (for
example, an execution problem), it is fixed and 12 weeks of paper are repeated.

## 2. Start

- **Initial capital: 1,000 USDT** *(proposal, to be confirmed; alternatives: 300 or 3,000)*.
  - About 42 USDT per leg, above Bitget's minimum (~5 USDT) and with little rounding error.
  - Maximum loss before the kill switch: ~350 USDT.
- **Server: VPS with a fixed IP** *(proposal, to be confirmed; alternative: this Mac)*.
  - The subaccount API key is bound to that IP.
  - Key permissions: futures only, read and trade; never withdrawals.
- **Steps** (first week after the gate is passed):
  1. Set up the VPS with the pinned commit, the monitor and the service keep-alive.
  2. Run the dry-run on the VPS for a week in parallel with the Mac, until both generate the
     same orders.
  3. Transfer the capital to the subaccount. The transfer is detected automatically; nothing
     needs adjusting.
  4. `init --mode live` and enable the job with `MODE=live LIVE_TRADING=1`.
  5. After the first rebalance: `report --mode live --paper-dir …` and a manual review of every
     order.
- The paper book **keeps running** in parallel as a reference for as long as the live strategy
  exists.

## 3. Scaling

**In stages** *(proposal, to be confirmed; alternatives: fixed for 12 weeks, or decide later)*.

Every 4 live rebalances, review:

| Check | Threshold |
|---|---|
| Actual slippage + fee, median per tier | ≤ model (3/6/12 bp slippage) |
| Daily correlation live vs paper | ≥ 0.9 |
| Rebalances missed or aborted | 0 |
| Risk state | neither `halted` nor `reduce_only` during the period |

- **All OK:** capital doubles (1k → 2k → 4k → 8k).
- **One fails:** capital stays the same.
- **Two consecutive reviews with failures:** drop one stage.
- **Initial cap: 10,000 USDT** *(proposal, to be confirmed)*. Raising it requires re-running
  `check_bitget_universe.py` at the new order size and confirming the book's cost stays within
  the model.

Scaling does **not** depend on realised return: four good or bad weeks are noise.

## 4. Stop criteria

**Automatic** (already implemented in the runner):
- **Drawdown ≥ 35% from peak:** kill switch as a **review pause**. It closes everything and
  stops until the review. `reset-risk` only after the review in §4.3; never on the same day.
- **Daily loss ≥ 8%:** reduce-only mode until the next UTC day.
- **Sanity limits:** weight > 10%, gross > 1x, order > 5,000 USDT or missing price. The
  rebalance is aborted.
- **Foreign positions or hedge mode:** abort.

**Strategy** (manual review, evaluated at each 4-week review):
- **Cumulative live return < −15%:** scaling stops and capital returns to the initial stage.
- **After 26 live weeks, live Sharpe < 0 and also below the paper over the same period:** the
  strategy is stopped and the hypothesis reviewed.
- **Live vs paper gap worse than −5%/yr sustained over 2 reviews:** stop and investigate
  execution.

**Kill switch threshold: −35%, as a pause** *(decided on 2026-10-21; previously −25%)*.
- The x1 backtest with hourly marks (`docs/plans/vol_targeting_hypothesis.md`) would have hit
  −25% on 2021-08-26 and on 2024-08-28. Its worst hourly drawdown was −26% in development and
  −35% out of sample.
- At −25%, a drawdown that is normal for the strategy would have stopped it before its best
  stretch.
- −35% is roughly the worst drawdown observed: hitting it means the strategy is outside what the
  backtest showed.

**Review after a kill switch:**
- **Market cause** (the strategy simply lost): it is a pause, not a retirement.
  - Compare the drawdown with the backtest's over the same period, running the backtest up to
    the date.
  - If comparable, resume with `reset-risk --confirm`, which resets the reference peak.
  - The capital stage remains governed by the cumulative-return < −15% rule.
  - If the drawdown is clearly worse than the backtest's, stop and review the hypothesis.
- **Implementation cause:** fix it, pass CI and dry-run before restarting.

## 5. Short-side tail risk

Observed example: BR rose +47% in one day while short in the 19-Oct simulation; it cost ~2% of
the portfolio at a weight of only 4.2%.

**Decision: accept and monitor** *(proposal, to be confirmed)*. Reasons:
- The tested hypothesis is not changed.
- The automatic guards (−8% daily, −35% total) bound the portfolio loss.
- **Per-position alert** (implemented on 2026-10-20): the monitor alerts once per position and
  rebalance when a single position loses more than 1.5% of equity since the last rebalance, both
  in paper and live. It only alerts; it does not trade.

The variants (a per-position stop, or not shorting tokens listed for less than 90 days) are
**new hypotheses**: they need their own backtest and their own paper before production. They are
not applied mid-test.

## 6. Operations

- **Every Monday** (~01:20 UTC): the monitor reports the rebalance. Review the notification,
  `status` and, when live, `report`.
- **Every 4 weeks:** scaling review (§3) and strategy criteria (§4).
- **Every 90 days:** rotate the subaccount API key.
- **Deposits and withdrawals:** only via Bitget internal transfer; the runner detects them.
- **Code changes:** branch → green CI → dry-run on the server → move the pinned commit. Never
  edit files in the production worktree or server.

## 7. Dates

| Date | Milestone |
|---|---|
| 2026-10-26 | First paper and dry-run rebalance |
| 2026-11-23 | Interim paper review (5 rebalances): operation only, no decisions |
| 2027-01-11 | Rebalance no. 12: gate review (§1) |
| ~2027-01-18 | Live start if the gate passes and the VPS is ready |
| ~2027-02-15 | First scaling review (4 live rebalances) |

## 8. Amendment of 2026-10-26: accelerated start

Written on 2026-10-26, after the paper's first rebalance and before the second. **Reason:** the
account owner wants the strategy in production within weeks rather than in January. It is not
motivated by results: at this date there are a few hours of P&L, which is noise.

Only **when** real money starts trading changes. The original 12-rebalance gate did not measure
profitability but whether the implementation reproduces the model (§1). Here that is verified in
less time, at the cost of more execution risk, which is bounded by small capital and by not
scaling until the original gate is met.

**Scope:** only the primary hypothesis (§0). Taker flow, the 10%-per-side variant and TSMOM stay
in paper with their own dates and rules, unchanged.

### Accelerated gate (decision on Monday 2026-11-09, after the fill)

**All** items must be met:

| Criterion | Threshold |
|---|---|
| Paper: rebalances executed / missed | 3 (2026-10-26, 11-02, 11-09) / 0 |
| Paper: median slippage per tier | ≤ 3 / 6 / 12 bp |
| Paper vs model | Same target positions as the replay; daily correlation ≥ 0.9 (indicative with few days) |
| Dry-run with the subaccount funded at initial capital | 2 rebalances (11-02 and 11-09) with orders generated |
| Dry-run: order errors, aborts, sanity aborts, unexplained Python errors | 0 |
| Dry-run vs paper | Same targets for the same decision_day |

**If an item fails:** no start. The cause is documented, fixed, and two dry-runs are repeated.

### Start

- **Initial capital: 300 USDT** (decided on 2026-10-26; replaces the proposal in §2).
  - ~12.5 USDT per leg. Weekly adjustments below ~5 USDT (Bitget's minimum) are skipped, so the
    live book will deviate slightly from the paper. Reviews against the paper must account for
    this.
  - Maximum loss before the kill switch: ~105 USDT.
- **Server: this Mac** (decided on 2026-10-26), with launchd, the monitor and keep-awake already
  installed. If it restarts or fails, the rebalance is retried every hour up to the delay limit.
  **A VPS is mandatory before going above 1,000 USDT.**
- **First live rebalance:** decision on Sunday 2026-11-15, execution on Monday 2026-11-16 at
  01:00 UTC.
- Steps 3–5 of §2 unchanged. The monitor adds the live book as another runner.

### Scaling

**No scaling** until the paper passes the original gate of §1 at the 2027-01-11 review. If it
passes, capital rises to 1,000 USDT and §3 applies from there. If it does not, §1 ("if a
criterion fails") applies to the live book: it is stopped and investigated.

### What does not change

- The stop criteria of §4: −35% kill switch as a pause, −8% daily, sanity limits.
- The strategy rules of §4 (−15% cumulative, 26-week Sharpe, gap vs paper).
- The paper keeps running as a reference.

### Dates

| Date | Milestone |
|---|---|
| Before 2026-11-01 | Fund the subaccount up to 300 USDT |
| 2026-11-02 | First dry-run with real orders generated |
| 2026-11-09 | Second dry-run and accelerated-gate decision |
| 2026-11-16 | First live rebalance (if approved) |
| 2027-01-11 | Original gate (§1) on the paper: condition for scaling to 1,000 USDT |
