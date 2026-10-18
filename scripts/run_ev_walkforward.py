"""Phase 2 of the EV score: walk-forward test of whether candidates can be ranked.

Scores compared out of sample (monthly folds, expanding window, purge on exit_ts + embargo):
  take_all       constant score (the bar every score must beat)
  cost_rule      -expected_cost_R: prefers wide stops / high volatility. Captures the purely
                 mechanical effect of costs, so a model must beat it to show real skill.
  model_net      GBM predicting net_R directly
  model_ev       GBM predicting gross_R, minus the known expected cost: E[gross R] - cost_R
  model_fwd12    GBM predicting the policy-agnostic 12-bar forward return (ATR units, net),
                 evaluated against that same target

Example:
  PYTHONPATH=src python scripts/run_ev_walkforward.py \
    --labels artifacts/ev_score/labels_v1_20261018/candidates_labeled.csv.gz
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
from hf_score.walkforward import evaluate_scores, run_walkforward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--first-test", default="2022-01-01")
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
    d["fwd_gross_atr_12"] = d["fwd_net_atr_12"] + rt_cost / d["atrp"]
    d["fwd_exit_dt"] = d["ts"] + pd.Timedelta(hours=12)

    X = build_model_frame(d, round_trip_cost=rt_cost)

    common = dict(ts=d["ts"], first_test=args.first_test, embargo_bars=args.embargo_bars)
    d["take_all"] = np.where(d["ts"] >= pd.Timestamp(args.first_test, tz="UTC"), 1.0, np.nan)
    d["cost_rule"] = np.where(d["take_all"].notna(), -X["expected_cost_R"], np.nan)
    d["model_net"] = run_walkforward(X, d["net_R"], exit_ts=d["exit_dt"], **common)
    pred_gross = run_walkforward(X, d["gross_R"], exit_ts=d["exit_dt"], **common)
    d["model_ev"] = pred_gross - X["expected_cost_R"]
    d["model_gross_only"] = pred_gross
    d["model_fwd12"] = run_walkforward(X, d["fwd_net_atr_12"], exit_ts=d["fwd_exit_dt"], target_clip=6.0, **common)

    report = {
        "policy_label": evaluate_scores(d, ["take_all", "cost_rule", "model_net", "model_ev", "model_gross_only"]),
        "forward_label_12": evaluate_scores(d, ["model_fwd12"], outcome="fwd_net_atr_12", gross="fwd_gross_atr_12"),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"artifacts/ev_score/walkforward_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["timestamp", "symbol", "strategy_id", "side", "net_R", "gross_R", "cost_R", "fwd_net_atr_12",
            "cost_rule", "model_net", "model_ev", "model_gross_only", "model_fwd12"]
    d.loc[d["take_all"].notna(), keep].to_csv(out_dir / "oos_predictions.csv.gz", index=False, compression="gzip")
    meta = {
        "created_utc": stamp,
        "labels": args.labels,
        "labels_manifest_created": manifest.get("created_utc"),
        "first_test": args.first_test,
        "embargo_bars": args.embargo_bars,
        "features": MODEL_FEATURES,
        "round_trip_cost": rt_cost,
        "seconds": round(time.perf_counter() - t0, 1),
        "report": report,
    }
    (out_dir / "walkforward_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {out_dir}  ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
