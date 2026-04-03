from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts" / "sweeps" / "exit_tp_trend"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_REGISTRY = ROOT / "artifacts" / "exit_policy_registry.json"
BACKUP_REGISTRY = ARTIFACTS_DIR / "exit_policy_registry.__base_backup__.json"
SUMMARY_CSV = ROOT / "results" / "sweep_exit_tp_trend_summary.csv"

MAX_CYCLES = 100

RUN_MATRIX = {
    "dot_trend":  [2.2, 2.4, 2.6, 2.8, 3.0],
    "eth_trend":  [2.4, 2.5, 2.7, 2.8, 3.0],
    "aave_trend": [2.5, 2.7, 2.9, 3.1],
    "bnb_trend":  [2.2, 2.4, 2.6, 2.8],
    "link_trend": [2.2, 2.4, 2.6, 2.8],
    "xrp_trend":  [2.0, 2.2, 2.4, 2.6],
}

BASE_ARGS = [
    sys.executable,
    "scripts/research_runtime.py",
    "--strategy-registry", "artifacts/rt_registry_add_trx_specialized.json",
    "--selection-policy-config", "artifacts/selection_policy_config.calibrated.json",
    "--selection-policy-profile", "research",
    "--policy-config", "artifacts/policy_config.json",
    "--policy-profile", "symmetric_v1",
    "--selection-semantics-mode", "research",
    "--start", "2025-04-01 00:00:00",
    "--end", "2026-04-01 00:00:00",
    "--exchange", "binanceusdm",
    "--cache-dir", "data/cache",
    "--allocator-mode", "production_like_snapshot",
    "--allocator-profile", "blended",
    "--projection-profile", "research",
    "--runtime-prod-ml-position-sizing",
    "--prodlike-allocator-apply-ml-sizing",
    "--runtime-ml-size-mode", "calibrated",
    "--runtime-ml-size-scale", "8.5",
    "--runtime-ml-size-min", "0.50",
    "--runtime-ml-size-max", "1.50",
    "--runtime-ml-size-base", "0.70",
    "--runtime-ml-size-pwin-threshold", "0.46",
    "--pwin-calibration-artifact", "artifacts/pwin_calibration_strategy_side_baseline_prodsem_3m_v3.json",
    "--target-exposure", "0.40",
    "--symbol-cap", "0.28",
]

METRIC_KEYS = [
    "total_return_pct",
    "sharpe_annual",
    "max_drawdown_pct",
    "trade_count",
    "win_rate_pct",
    "avg_active_symbols",
    "avg_gross_weight",
    "rows",
    "load_seconds",
    "run_seconds",
    "total_seconds",
]

SUMMARY_HEADER = [
    "cycle",
    "run_name",
    "strategy_id",
    "tp_atr_mult",
    "status",
    "metrics_path",
    "registry_path",
] + METRIC_KEYS


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def ensure_summary_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(SUMMARY_HEADER)


def append_summary(path: Path, row: dict) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        writer.writerow({k: row.get(k, "") for k in SUMMARY_HEADER})


def build_jobs() -> list[tuple[str, float]]:
    jobs: list[tuple[str, float]] = []
    for strategy_id, vals in RUN_MATRIX.items():
        for tp_atr_mult in vals:
            jobs.append((strategy_id, float(tp_atr_mult)))
    return jobs[:MAX_CYCLES]


def make_registry_variant(base_data: dict, strategy_id: str, tp_atr_mult: float) -> Path:
    data = json.loads(json.dumps(base_data))
    if "strategies" not in data or strategy_id not in data["strategies"]:
        raise KeyError(f"Strategy not found in registry: {strategy_id}")

    data["strategies"][strategy_id]["params"]["tp_atr_mult"] = float(tp_atr_mult)

    out = ARTIFACTS_DIR / f"exit_policy_registry__{strategy_id}__tp_{str(tp_atr_mult).replace('.', '_')}.json"
    save_json(out, data)
    return out


def restore_registry() -> None:
    if BACKUP_REGISTRY.exists():
        shutil.copy2(BACKUP_REGISTRY, BASE_REGISTRY)


def main() -> None:
    if not BASE_REGISTRY.exists():
        raise SystemExit(f"No existe registry base: {BASE_REGISTRY}")

    ensure_summary_header(SUMMARY_CSV)

    shutil.copy2(BASE_REGISTRY, BACKUP_REGISTRY)
    base_data = load_json(BASE_REGISTRY)
    jobs = build_jobs()

    print(f"Total cycles to run: {len(jobs)}")
    print(f"Max cycles allowed: {MAX_CYCLES}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Base registry: {BASE_REGISTRY}")
    print()

    try:
        for cycle, (strategy_id, tp_atr_mult) in enumerate(jobs, start=1):
            stamp = time.strftime("%Y%m%dT%H%M%S")
            run_name = f"sweep_tp_{strategy_id}_{str(tp_atr_mult).replace('.', '_')}_{stamp}"
            registry_path = make_registry_variant(base_data, strategy_id, tp_atr_mult)

            shutil.copy2(registry_path, BASE_REGISTRY)

            cmd = list(BASE_ARGS)
            cmd.extend(["--name", run_name])

            print("=" * 120)
            print(f"CYCLE {cycle}/{len(jobs)}")
            print(f"RUN: {run_name}")
            print(f"strategy_id={strategy_id} tp_atr_mult={tp_atr_mult}")
            print(f"registry_variant={registry_path}")
            print(f"active_registry={BASE_REGISTRY}")
            print("CMD:", " ".join(cmd))
            print("=" * 120)

            started = time.time()
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
            )
            elapsed = time.time() - started

            metrics_path = ROOT / "results" / f"research_runtime_metrics_{run_name}.json"

            row = {
                "cycle": cycle,
                "run_name": run_name,
                "strategy_id": strategy_id,
                "tp_atr_mult": tp_atr_mult,
                "status": "ok" if proc.returncode == 0 and metrics_path.exists() else f"failed_rc_{proc.returncode}",
                "metrics_path": str(metrics_path) if metrics_path.exists() else "",
                "registry_path": str(registry_path),
                "run_seconds": elapsed,
            }

            if metrics_path.exists():
                try:
                    metrics = load_json(metrics_path)
                    for k in METRIC_KEYS:
                        if k in metrics:
                            row[k] = metrics.get(k)
                except Exception as e:
                    row["status"] = f"metrics_parse_error: {e}"

            append_summary(SUMMARY_CSV, row)
            restore_registry()

            print(f"APPENDED: {row['status']} {row['run_name']}")
            print(f"END CYCLE {cycle}/{len(jobs)}")
            print()

    finally:
        restore_registry()

    print("DONE")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Registry restored to: {BASE_REGISTRY}")


if __name__ == "__main__":
    main()
