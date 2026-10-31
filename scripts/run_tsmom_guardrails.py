"""Exploratory: how the live runner's guardrails change the BTC/ETH TSMOM book
(docs/plans/tsmom_guardrails_exploration.md). Decides nothing for the registered paper test.

Hour-by-hour account with the live rules of hf_research.live_xs.check_risk: kill switch on the
drawdown from peak (flatten, then pause), daily-loss reduce-only measured from the first run of
the UTC day, a cap per asset, and the rebalance band. Targets are the primary TsmomConfig().

  PYTHONPATH=src python scripts/run_tsmom_guardrails.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from hf_research.tsmom import TsmomConfig, tsmom_targets
from hf_research.vol_target import load_hourly_close
from hf_research.xs_momentum import load_daily_panels

ASSETS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
IS_START, OOS_START, END = "2020-04-01", "2024-01-01", "2026-10-19"
COST = 0.0006 + 3.0 / 10000.0


@dataclass(frozen=True)
class Guards:
    kill_dd: float | None = 0.35  # None = no kill switch
    pause_days: int | None = 7  # None = stop for good after a kill
    daily_loss: float | None = 0.08
    daily_rule: str = "utc_day"  # utc_day (live runner) | rolling_24h
    max_weight: float = 1.0
    band: float = 0.05


def simulate(targets: pd.DataFrame, hourly: pd.DataFrame, funding: pd.DataFrame, g: Guards, start: str, end: str) -> dict:
    """Equity path from 1 at `start`. Decision day d fills at the close of the 00:00 bar of d+1."""
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    px = hourly[ASSETS].loc[t0:t1 + pd.Timedelta(days=1)].ffill()
    times, p = px.index, px.to_numpy()
    w_tgt = targets[ASSETS].clip(-g.max_weight, g.max_weight)
    fund = funding[ASSETS].fillna(0.0)
    cash, q = 1.0, np.zeros(len(ASSETS))
    peak, halted_until, stopped = 1.0, None, False
    day_start, kills, daily_eq, eq_hist = 1.0, 0, {}, []
    worst_dd = 0.0
    for k, t in enumerate(times):
        mark = p[k]
        eq = cash + float(q @ mark)
        eq_hist.append(eq)
        if t.hour == 0:  # first hourly run of the UTC day (the runner's day_start)
            day_start = eq
        # Funding for row d-1 is charged at the day-d fill (engine convention), here at 00:00 bars.
        if t.hour == 0:
            d = (t - pd.Timedelta(days=1)).normalize()
            if d in fund.index:
                cash -= float((q * mark) @ fund.loc[d].to_numpy())
                eq = cash + float(q @ mark)
        peak = max(peak, eq)
        worst_dd = min(worst_dd, eq / peak - 1.0)
        halted = stopped or (halted_until is not None and t < halted_until)
        if not halted and halted_until is not None:
            halted_until, peak = None, eq  # resume after the review with the peak reset
        if not halted and g.kill_dd is not None and eq <= peak * (1.0 - g.kill_dd) and np.any(q):
            cash -= float(np.abs(q) @ mark) * COST
            cash += float(q @ mark)
            q[:] = 0.0
            kills += 1
            if g.pause_days is None:
                stopped = True
            else:
                halted_until = t + pd.Timedelta(days=g.pause_days)
            continue
        if t.hour == 0 and not halted:  # fill of decision day t-1, at this bar's close (01:00)
            d = (t - pd.Timedelta(days=1)).normalize()
            if d in w_tgt.index and t0 <= d:
                fill = mark if k + 1 >= len(times) else p[k + 1]
                eq_f = cash + float(q @ fill)
                cur = q * fill / eq_f
                tgt = w_tgt.loc[d].to_numpy().copy()
                same = (cur != 0) & (tgt != 0) & (np.sign(cur) == np.sign(tgt)) & (np.abs(tgt - cur) < g.band)
                tgt[same] = cur[same]
                if g.daily_loss is not None:
                    ref = day_start if g.daily_rule == "utc_day" else (eq_hist[-24] if len(eq_hist) >= 24 else 1.0)
                    if 1.0 - eq / ref >= g.daily_loss:
                        tgt = np.where(np.sign(tgt) == np.sign(cur), np.sign(cur) * np.minimum(np.abs(tgt), np.abs(cur)), 0.0)
                new_q = tgt * eq_f / fill
                cash -= float(np.abs(new_q - q) @ fill) * COST
                cash -= float((new_q - q) @ fill)
                q = new_q
        if t.hour == 1:
            daily_eq[t.normalize()] = eq
    s = pd.Series(daily_eq)
    s = s[(s.index >= t0) & (s.index < t1)]
    r = s.pct_change().dropna()
    years = len(r) / 365.0
    return {"cagr": float(s.iloc[-1] ** (1 / years) - 1) if s.iloc[-1] > 0 else -1.0,
            "sharpe": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else float("nan"),
            "max_dd_hourly": float(worst_dd), "final_equity": float(s.iloc[-1]), "kills": kills}


VARIANTS = {
    "no_guards": Guards(kill_dd=None, daily_loss=None),
    "current (kill 35%, pause 7d, daily 8%)": Guards(),
    **{f"kill {int(k * 100)}%, pause 7d": Guards(kill_dd=k) for k in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50)},
    **{f"kill {int(k * 100)}%, pause 30d": Guards(kill_dd=k, pause_days=30) for k in (0.20, 0.25, 0.35)},
    **{f"kill {int(k * 100)}%, stop for good": Guards(kill_dd=k, pause_days=None) for k in (0.25, 0.35)},
    **{f"daily {int(x * 100)}% (live rule)": Guards(daily_loss=x) for x in (0.03, 0.05, 0.12)},
    **{f"daily {int(x * 100)}% (rolling 24h)": Guards(daily_loss=x, daily_rule="rolling_24h") for x in (0.03, 0.05, 0.08)},
    **{f"max weight {w}": Guards(max_weight=w) for w in (0.25, 0.5, 0.75, 1.5)},
    **{f"band {b}": Guards(band=b) for b in (0.0, 0.02, 0.10, 0.20)},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="artifacts/research/tsmom_guardrails_v1")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panels = load_daily_panels(ASSETS, cache_dir=args.cache_dir)
    cfg = TsmomConfig()
    raw = tsmom_targets(panels["signal_close"], replace(cfg, max_weight=10.0))  # cap applied per variant
    hourly = load_hourly_close(ASSETS, args.cache_dir)
    rows = []
    for name, g in VARIANTS.items():
        row = {"variant": name}
        for p, (a, b) in {"is": (IS_START, OOS_START), "oos": (OOS_START, END), "full": (IS_START, END)}.items():
            for k, v in simulate(raw, hourly, panels["funding"], g, a, b).items():
                row[f"{p}_{k}"] = v
        rows.append(row)
        print(f"{name:<42} dev SR {row['is_sharpe']:+.2f}  oos SR {row['oos_sharpe']:+.2f}  "
              f"oos CAGR {row['oos_cagr']:+.1%}  DD dev/oos {row['is_max_dd_hourly']:.0%}/{row['oos_max_dd_hourly']:.0%}  "
              f"kills {row['full_kills']}  full eq {row['full_final_equity']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(out / "guardrails.csv", index=False)
    (out / "variants.json").write_text(json.dumps({k: v.__dict__ for k, v in VARIANTS.items()}, indent=2))


if __name__ == "__main__":
    main()
