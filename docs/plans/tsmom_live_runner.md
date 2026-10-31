# Live runner for TSMOM BTC/ETH (infrastructure, not a decision)

*English translation. The Spanish original is in [`es/`](es/tsmom_live_runner.md).*

Written on 2026-10-31. It prepares the infrastructure so that, **if** TSMOM or the combination
passes on 2027-04-19 (`tsmom_paper_plan.md`), it can start without building anything at the last
minute. **It does not change the registered plan:** TSMOM does not trade live before that
decision, and the decision does not depend on this runner.

## What it is

`scripts/live_tsmom.py` (wrapper `deploy/run_live_tsmom.sh`):
- **Signal, schedule and band:** the same as the paper. It uses the same target function as
  `scripts/paper_tsmom.py`, decides each day *d* and executes on *d*+1 at 01:00 UTC. A fill more
  than 12 h late is recorded as missed.
- **Controls:** the same as the cross-sectional runner (`hf_research.live_xs`):
  - kill switch at a −35% drawdown from peak: closes everything and pauses until `reset-risk`;
  - −8% on the day: reduce-only mode;
  - sanity limits that abort the rebalance;
  - transfers do not count as P&L;
  - aborts on positions it did not open.
- **Modes:** `dry-run` (default, sends nothing) and `live`, which additionally requires
  `LIVE_TRADING=1`.

**Differences from the cross-sectional configuration:**

| Parameter | XS | TSMOM | Reason |
|---|---|---|---|
| Maximum weight per asset | 0.10 | 1.0 | `TsmomConfig.max_weight` |
| Maximum gross exposure | 1.0 | 2.0 | two assets of up to 1.0 each |
| Maximum fill delay | 24 h | 12 h | same as the paper |
| Reported post-trade drift | 1% | 3% | ETH trades in 0.01 steps (~22 USDT) |
| Per-position loss alert | 1.5% | 3% | positions of ~20% of capital each |

## Its own subaccount (mandatory)

The cross-sectional book also trades BTC and ETH, and each runner aborts on the other's positions.
So TSMOM:
- reads credentials **only** from `~/.config/hedgefund/bitget_tsmom.env` (or
  `$BITGET_TSMOM_ENV_FILE`), with chmod 600 and a key without withdrawal permission;
- never uses the cross-sectional file or the `BITGET_*` variables;
- refuses a key identical to the cross-sectional one.

## Minimum capital

Bitget's ETH minimum is 0.01 (~22 USDT). With the test day 2026-10-29 (BTC +0.186, ETH −0.162),
the ETH order is rounded to:

| Capital | ETH target | ETH sent | Rounding error |
|---|---|---|---|
| 300 | 48.6 USDT | 0.02 (44.9) | −8% |
| 500 | 81.0 USDT | 0.03 (67.4) | −17% |
| 1,000 | 162 USDT | 0.07 (157) | −3% |
| 2,000 | 324 USDT | 0.14 (315) | −3% |

Also, daily adjustments that clear the band are 5% of capital or more, and must exceed ~22 USDT in
ETH. With less than ~500 USDT many are skipped, and the book drifts from the paper.
**Recommended capital: ≥ 1,000 USDT** in the TSMOM subaccount.

## Steps to activate it (when the account owner decides)

1. Create a new Bitget subaccount and a trade-only API key, without withdrawal. Store it in
   `~/.config/hedgefund/bitget_tsmom.env` with chmod 600.
2. Dry-run: `init --mode dry-run`, a launchd agent at minute 17 with `MODE=dry-run` and the runner
   in the health check (`--runner tsmom-dry-run=<state>:<logs>`). Run it for a few weeks and check
   that its targets match the paper every day.
3. Live, only if the 2027-04-19 decision approves it and the owner confirms:
   - `init --mode live`;
   - agent with `MODE=live LIVE_TRADING=1`;
   - capital per the cross-sectional scaling plan;
   - VPS if the total exceeds 1,000 USDT.
