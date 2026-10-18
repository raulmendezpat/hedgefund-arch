"""Does a per-asset ML discriminator rank candidates better than one global model?

All variants predict gross_R and score = prediction - expected_cost_R (the EV design):
  ev_global         one model for every asset
  ev_global_symcat  one model with the symbol as a categorical feature
  ev_per_symbol     an independent model per symbol
  ev_per_sym_side   an independent model per symbol and side
  cost_rule         -expected_cost_R, the mechanical baseline every model must beat

Besides pooled metrics it reports, per symbol, the monthly IC against gross_R split into two
halves of the out-of-sample period, so a per-asset "edge" that does not persist is visible
as such (with 20 assets some will look good in one half by chance).

Example:
  PYTHONPATH=src python scripts/run_ev_asset_discriminator.py \
    --labels artifacts/ev_score/labels_v2_20261018/candidates_labeled.csv.gz
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hf_score.features import MODEL_FEATURES, build_model_frame
from hf_score.walkforward import (
    categorical_model,
    evaluate_scores,
    monthly_ic,
    run_walkforward,
    run_walkforward_grouped,
)

SCORES = ["cost_rule", "ev_global", "ev_global_symcat", "ev_per_symbol", "ev_per_sym_side"]


def per_symbol_report(d: pd.DataFrame, score: str, split: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for sym, g in d[d[score].notna()].groupby("symbol"):
        first, second = g[g["ts"] < split], g[g["ts"] >= split]
        ic1 = monthly_ic(first, score, "gross_R", min_rows=30)
        ic2 = monthly_ic(second, score, "gross_R", min_rows=30)
        q = pd.qcut(g[score].rank(method="first"), 5, labels=False)
        top = g[q == 4]
        rows.append(
            {
                "symbol": sym,
                "rows": len(g),
                "ic_gross_h1": ic1.mean(),
                "ic_gross_h1_pos": (ic1 > 0).mean(),
                "ic_gross_h2": ic2.mean(),
                "ic_gross_h2_pos": (ic2 > 0).mean(),
                "top_q_net_R": top["net_R"].mean(),
                "top_q_gross_R": top["gross_R"].mean(),
                "take_gt0_n": int((g[score] > 0).sum()),
                "take_gt0_net_R": g.loc[g[score] > 0, "net_R"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("ic_gross_h2", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--first-test", default="2022-01-01")
    ap.add_argument("--split", default="2024-07-01", help="Boundary between the two OOS halves in the per-symbol report")
    ap.add_argument("--embargo-bars", type=int, default=48)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    t0 = time.perf_counter()
    manifest = json.loads((Path(args.labels).parent / "manifest.json").read_text(encoding="utf-8"))
    rt_cost = float(manifest["label_config"]["round_trip_cost"])

    d = pd.read_csv(args.labels, low_memory=False)
    d = d[d["label_complete"] == True].reset_index(drop=True)  # noqa: E712
    d["ts"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
    d["exit_dt"] = pd.to_datetime(d["exit_ts"], unit="ms", utc=True)
    d["cost_R"] = rt_cost / d["risk_frac"]
    d["gross_R"] = d["net_R"] + d["cost_R"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"artifacts/ev_score/asset_discriminator_{stamp}")

    X = build_model_frame(d, round_trip_cost=rt_cost)
    cost = X["expected_cost_R"]
    common = dict(first_test=args.first_test, embargo_bars=args.embargo_bars)
    in_oos = d["ts"] >= pd.Timestamp(args.first_test, tz="UTC")

    d["cost_rule"] = np.where(in_oos, -cost, np.nan)
    d["ev_global"] = run_walkforward(X, d["gross_R"], d["ts"], d["exit_dt"], **common) - cost

    Xc = X.copy()
    Xc["symbol_code"] = d["symbol"].astype("category").cat.codes.astype(float)
    d["ev_global_symcat"] = (
        run_walkforward(Xc, d["gross_R"], d["ts"], d["exit_dt"], model_factory=categorical_model(Xc.columns.get_loc("symbol_code")), **common)
        - cost
    )
    d["ev_per_symbol"] = run_walkforward_grouped(X, d["gross_R"], d["ts"], d["exit_dt"], d["symbol"], **common) - cost
    d["ev_per_sym_side"] = (
        run_walkforward_grouped(X, d["gross_R"], d["ts"], d["exit_dt"], d["symbol"] + "|" + d["side"], **common) - cost
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    d.loc[in_oos, ["timestamp", "symbol", "side", "net_R", "gross_R", *SCORES]].to_csv(
        out_dir / "oos_predictions.csv.gz", index=False, compression="gzip"
    )

    # Compare every score on the same rows (the per-asset models skip assets/folds with short history).
    common_rows = d[SCORES].notna().all(axis=1)
    pooled = evaluate_scores(d[common_rows], SCORES)

    split = pd.Timestamp(args.split, tz="UTC")
    per_sym = {}
    for s in ["ev_global", "ev_per_symbol", "ev_per_sym_side"]:
        rep = per_symbol_report(d[common_rows], s, split)
        rep.to_csv(out_dir / f"per_symbol_{s}.csv", index=False)
        per_sym[s] = {
            "symbols_ic_gross_pos_both_halves": int(((rep["ic_gross_h1"] > 0) & (rep["ic_gross_h2"] > 0)).sum()),
            "symbols_top_q_net_positive": int((rep["top_q_net_R"] > 0).sum()),
            "symbols": int(len(rep)),
        }

    meta = {
        "created_utc": stamp,
        "labels": args.labels,
        "first_test": args.first_test,
        "split": args.split,
        "embargo_bars": args.embargo_bars,
        "features": MODEL_FEATURES,
        "rows_common": int(common_rows.sum()),
        "pooled": pooled,
        "per_symbol_summary": per_sym,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "asset_discriminator_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"rows_common": meta["rows_common"], "per_symbol_summary": per_sym}, indent=2))
    for s, r in pooled.items():
        print(f"{s:18s} ic_net={r['ic_net_mean']:.3f} ({r['ic_net_pos_share']:.0%}) ic_gross={r['ic_gross_mean']:.3f} "
              f"({r['ic_gross_pos_share']:.0%}) q_net={r['quintile_net_R']} take>0 n={r['take_score_gt_0']['n']} "
              f"net={r['take_score_gt_0']['mean_net_R']:.3f} months+={r['take_score_gt_0']['months_positive_share']}")
    print(f"saved: {out_dir} ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
