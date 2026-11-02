"""Exploratory: model and pipeline changes that could raise return without much more risk
(docs/plans/pipeline_ideas.md). Decides nothing for the registered papers.

Ideas:
  A. funding filter (XS, frac10): drop a selected long whose trailing 7-day funding is above +x/yr,
     or a selected short below -x/yr, and let the next-ranked asset take its place; x = 50/100/200%.
  B. maker execution (all books): fee 0.02% and no slippage, or half of the volume as maker.
     An upper bound only: unfilled orders and adverse selection are not modelled.
  C. combining books (equal capital, daily rebalanced): XS+TSMOM, XS+taker, XS+TSMOM+taker, and the
     same combinations scaled to the XS book's development-period volatility.

Screening rule, fixed before running: an idea is worth registering for paper only if, in
development AND out of sample, CAGR improves >= 2 points/yr, Sharpe does not fall, and the worst
drawdown is at most 3 points worse. For A, all three thresholds must pass and the result must hold
with costs x2 (a single threshold that passes is a fitted value, not a rule). B is reported as a
bound, never registered from a backtest.

  PYTHONPATH=src python scripts/run_pipeline_ideas.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_leverage_study import OOS_START, END, strategies  # noqa: E402

from hf_research.engine import backtest_weights, summarize  # noqa: E402
from hf_research.funding_carry import signal_funding, trailing_funding_ann  # noqa: E402
from hf_research.paper_xs import PRIMARY_STRATEGY  # noqa: E402
from hf_research.xs_momentum import MomentumConfig, eligible_from_membership, target_weights  # noqa: E402

RULE = {"cagr_gain_min": 0.02, "dd_worse_max": 0.03}
FUNDING_X = [0.5, 1.0, 2.0]
IS_START = "2021-03-01"


def as_daily(net: pd.Series) -> pd.DataFrame:
    """A combined book's net returns in the engine's daily layout (always active)."""
    return pd.DataFrame({"net": net, "gross_exposure": 1.0, "turnover": 0.0, "gross": 0.0, "funding": 0.0, "cost": 0.0})


def stats(df: pd.DataFrame, is_start: str = IS_START) -> dict:
    """Engine summary per period; days before a book first trades are left out, as in summarize."""
    out = {}
    for p, (a, b) in {"is": (is_start, OOS_START), "oos": (OOS_START, END)}.items():
        s = summarize(df, a, b)
        out.update({f"{p}_cagr": s.get("cagr"), f"{p}_sharpe": s.get("sharpe"), f"{p}_max_dd": s.get("max_dd"),
                    f"{p}_vol": s.get("vol")})
    return out


def run_book(s: dict, targets=None, fee=0.0006, slip_mult=1.0) -> pd.DataFrame:
    panels = dict(s["panels"])
    slip = panels.get("slippage_bps")
    if slip is not None:
        panels["slippage_bps"] = slip * slip_mult
    daily = backtest_weights(panels, s["targets"] if targets is None else targets, fee=fee,
                             slippage_bps=s.get("slippage_bps", 3.0) * slip_mult, band=s["band"],
                             rebalance_days=s["rebalance_days"])
    return daily


def funding_filtered(signal_close, cfg, members, fund_ann, x) -> pd.DataFrame:
    """Two passes: selected legs paying more than x/yr of trailing funding are made ineligible."""
    el = members.copy()
    for _ in range(2):
        w = target_weights(signal_close, cfg, el)
        f = fund_ann.reindex(index=w.index, columns=w.columns)
        bad = ((w > 0) & (f > x)) | ((w < 0) & (f < -x))
        if not bad.values.any():
            break
        el = el & ~bad
    return target_weights(signal_close, cfg, el)


def verdict(row: dict, base: dict) -> dict:
    ok = {}
    for p in ("is", "oos"):
        ok[p] = (row[f"{p}_cagr"] - base[f"{p}_cagr"] >= RULE["cagr_gain_min"]
                 and row[f"{p}_sharpe"] >= base[f"{p}_sharpe"]
                 and row[f"{p}_max_dd"] >= base[f"{p}_max_dd"] - RULE["dd_worse_max"])
    return {"passes_both": ok["is"] and ok["oos"], **{f"passes_{p}": v for p, v in ok.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", default="artifacts/research/universe_top60_v1/membership.csv")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-dir", default="artifacts/research/pipeline_ideas_v1")
    args = ap.parse_args()
    t0 = time.perf_counter()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    membership = pd.read_csv(args.membership)
    S = strategies(membership, args.cache_dir)
    rows, nets = [], {}

    def add(book, idea, daily, base=None, is_start=IS_START):
        r = {"book": book, "idea": idea, **stats(daily, is_start)}
        if base is not None:
            r.update(verdict(r, base))
        rows.append(r)
        nets[f"{book}|{idea}"] = daily["net"]
        print(f"{book:<12} {idea:<34} dev CAGR {r['is_cagr']:+.1%} SR {r['is_sharpe']:+.2f} DD {r['is_max_dd']:.0%} | "
              f"oos CAGR {r['oos_cagr']:+.1%} SR {r['oos_sharpe']:+.2f} DD {r['oos_max_dd']:.0%}"
              + (f"  {'PASS' if r.get('passes_both') else ''}" if base is not None else ""), flush=True)
        return r

    base = {n: add(n, "current", run_book(s)) for n, s in S.items()}
    base_x2 = {n: stats(run_book(s, fee=0.0012, slip_mult=2.0)) for n, s in S.items() if n.startswith("xs")}

    # A. funding filter
    perps = list(S["xs_momentum"]["panels"]["signal_close"].columns)
    idx = S["xs_momentum"]["panels"]["signal_close"].index
    fund_ann = trailing_funding_ann(signal_funding(perps, idx, args.cache_dir), 7)
    members = eligible_from_membership(membership, idx, perps)
    for book, frac in (("xs_momentum", 0.2), ("xs_frac10", 0.1)):
        cfg = MomentumConfig(**{**PRIMARY_STRATEGY, "select_frac": frac})
        for x in FUNDING_X:
            w = funding_filtered(S[book]["panels"]["signal_close"], cfg, members, fund_ann, x)
            r = add(book, f"A funding filter {int(x * 100)}%/yr", run_book(S[book], w), base[book])
            x2 = stats(run_book(S[book], w, fee=0.0012, slip_mult=2.0))
            r["x2_beats_current_x2"] = all(x2[f"{p}_cagr"] > base_x2[book][f"{p}_cagr"] for p in ("is", "oos"))

    # B. maker execution bounds
    for n, s in S.items():
        add(n, "B maker bound (fee 2bp, no slip)", run_book(s, fee=0.0002, slip_mult=0.0), base[n])
        add(n, "B half maker (fee 4bp, half slip)", run_book(s, fee=0.0004, slip_mult=0.5), base[n])

    # C. combinations, from the taker book's first development day (2021-12-01) so every leg trades
    c0 = "2021-12-01"
    xs, ts, tk = nets["xs_momentum|current"], nets["tsmom|current"], nets["taker_flow|current"]
    combos = {"C XS+TSMOM": (xs + ts) / 2, "C XS+taker": (xs + tk) / 2, "C XS+TSMOM+taker": (xs + ts + tk) / 3}
    xs_c = add("xs_momentum", "current from 2021-12", as_daily(xs[xs.index >= c0]), is_start=c0)
    for name, c in combos.items():
        c = c[c.index >= c0].dropna()
        add("combo", name, as_daily(c), xs_c, is_start=c0)
        k = xs_c["is_vol"] / (c.loc[c.index < OOS_START].std() * np.sqrt(365))
        r = add("combo", f"{name} at XS dev vol (x{k:.2f})", as_daily(c * k), xs_c, is_start=c0)
        r["scale"] = float(k)
    both = pd.concat([xs.rename("xs"), ts.rename("tsmom"), tk.rename("taker")], axis=1).dropna()
    corr = {p: both.loc[(both.index >= a) & (both.index < b)].corr().round(3).to_dict()
            for p, (a, b) in {"is": ("2021-12-01", OOS_START), "oos": (OOS_START, END)}.items()}

    pd.DataFrame(rows).to_csv(out / "ideas.csv", index=False)
    (out / "report.json").write_text(json.dumps({"rule": RULE, "funding_x": FUNDING_X, "daily_corr": corr,
                                                 "seconds": round(time.perf_counter() - t0, 1)}, indent=2))
    print(json.dumps(corr, indent=1))


if __name__ == "__main__":
    main()
