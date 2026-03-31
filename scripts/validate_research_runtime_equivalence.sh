#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/raulmendez/Projects/hedgefund-arch"
cd "$ROOT_DIR"

BASELINE_3M="results/research_runtime_metrics_rt_add_trx_specialized_revert_step7_3m.json"
BASELINE_6M="results/research_runtime_metrics_rt_add_trx_specialized_revert_step7_6m.json"

OUT_NAME_3M="rt_revert_step7_refactor_score_overrides_3m"
OUT_NAME_6M="rt_revert_step7_refactor_score_overrides_6m"

echo
echo "============================================================"
echo "VALIDACIÓN DE EQUIVALENCIA FUNCTIONAL"
echo "Repo: $ROOT_DIR"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git log --oneline -n 1)"
echo "============================================================"
echo

if [[ ! -f "$BASELINE_3M" ]]; then
  echo "ERROR: no existe baseline 3m: $BASELINE_3M"
  exit 1
fi

if [[ ! -f "$BASELINE_6M" ]]; then
  echo "ERROR: no existe baseline 6m: $BASELINE_6M"
  exit 1
fi

echo "==> Detectando comando baseline 3m desde shell history"
BASE_CMD_3M="$(fc -ln -300 | grep 'scripts/research_runtime.py' | grep 'rt_add_trx_specialized_revert_step7_3m' | tail -n 1 || true)"
if [[ -z "$BASE_CMD_3M" ]]; then
  echo "ERROR: no pude encontrar en tu history el comando de 3m con nombre rt_add_trx_specialized_revert_step7_3m"
  echo "Corre primero tu comando baseline 3m o pega aquí el comando exacto y te lo dejo fijo."
  exit 1
fi

echo "==> Detectando comando baseline 6m desde shell history"
BASE_CMD_6M="$(fc -ln -300 | grep 'scripts/research_runtime.py' | grep 'rt_add_trx_specialized_revert_step7_6m' | tail -n 1 || true)"
if [[ -z "$BASE_CMD_6M" ]]; then
  echo "ERROR: no pude encontrar en tu history el comando de 6m con nombre rt_add_trx_specialized_revert_step7_6m"
  echo "Corre primero tu comando baseline 6m o pega aquí el comando exacto y te lo dejo fijo."
  exit 1
fi

RUN_CMD_3M="${BASE_CMD_3M/rt_add_trx_specialized_revert_step7_3m/$OUT_NAME_3M}"
RUN_CMD_6M="${BASE_CMD_6M/rt_add_trx_specialized_revert_step7_6m/$OUT_NAME_6M}"

echo
echo "==> Comando 3m detectado:"
echo "$RUN_CMD_3M"
echo
echo "==> Comando 6m detectado:"
echo "$RUN_CMD_6M"

echo
echo "==> Ejecutando corrida 3m"
eval "$RUN_CMD_3M"

echo
echo "==> Ejecutando corrida 6m"
eval "$RUN_CMD_6M"

NEW_METRICS_3M="results/research_runtime_metrics_${OUT_NAME_3M}.json"
NEW_METRICS_6M="results/research_runtime_metrics_${OUT_NAME_6M}.json"

if [[ ! -f "$NEW_METRICS_3M" ]]; then
  echo
  echo "ERROR: no se generó metrics 3m esperado:"
  echo "  $NEW_METRICS_3M"
  exit 1
fi

if [[ ! -f "$NEW_METRICS_6M" ]]; then
  echo
  echo "ERROR: no se generó metrics 6m esperado:"
  echo "  $NEW_METRICS_6M"
  exit 1
fi

echo
echo "==> Comparando métricas contra baseline"
python - <<'PY'
import json
import sys
from pathlib import Path

baseline_3m = Path("results/research_runtime_metrics_rt_add_trx_specialized_revert_step7_3m.json")
baseline_6m = Path("results/research_runtime_metrics_rt_add_trx_specialized_revert_step7_6m.json")
new_3m = Path("results/research_runtime_metrics_rt_revert_step7_refactor_score_overrides_3m.json")
new_6m = Path("results/research_runtime_metrics_rt_revert_step7_refactor_score_overrides_6m.json")

keys_required = [
    "equity_final",
    "total_return_pct",
    "sharpe_annual",
    "max_drawdown_pct",
    "trade_count",
    "win_rate_pct",
    "avg_n_accepts",
    "avg_active_symbols",
    "avg_gross_weight",
]

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def compare_exact(label: str, base: dict, new: dict) -> list[str]:
    errors = []
    print(f"\n===== {label} =====")
    for k in keys_required:
        if k not in base:
            errors.append(f"[{label}] falta key en baseline: {k}")
            continue
        if k not in new:
            errors.append(f"[{label}] falta key en nuevo: {k}")
            continue

        b = base[k]
        n = new[k]

        print(f"{k}: baseline={b!r} | nuevo={n!r}")

        if isinstance(b, int) and isinstance(n, int):
            if b != n:
                errors.append(f"[{label}] {k} difiere: baseline={b!r} nuevo={n!r}")
        elif isinstance(b, (int, float)) and isinstance(n, (int, float)):
            if float(b) != float(n):
                errors.append(f"[{label}] {k} difiere: baseline={b!r} nuevo={n!r}")
        else:
            if b != n:
                errors.append(f"[{label}] {k} difiere: baseline={b!r} nuevo={n!r}")
    return errors

b3 = load_json(baseline_3m)
b6 = load_json(baseline_6m)
n3 = load_json(new_3m)
n6 = load_json(new_6m)

errors = []
errors.extend(compare_exact("3m", b3, n3))
errors.extend(compare_exact("6m", b6, n6))

print("\n============================================================")
if errors:
    print("RESULTADO: FALLÓ LA EQUIVALENCIA")
    print("------------------------------------------------------------")
    for e in errors:
        print(e)
    print("============================================================")
    sys.exit(1)
else:
    print("RESULTADO: EQUIVALENCIA EXACTA OK")
    print("============================================================")
PY

echo
echo "Validación finalizada correctamente."
