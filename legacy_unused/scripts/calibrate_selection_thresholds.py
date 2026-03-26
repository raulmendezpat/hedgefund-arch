from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


CANDIDATES_CSV = "results/research_runtime_candidates_diag_6m_meta_patch_1.csv"
REGISTRY_JSON = "artifacts/pwin_ml/pwin_ml_registry.json"
OUT_JSON = "artifacts/selection_policy_config.calibrated.json"


def load_registry(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_models(registry: dict) -> dict:
    out = {}
    models = registry.get("models", {}) or {}
    for symbol, meta in models.items():
        path = meta.get("path")
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        out[symbol] = {
            "bundle": joblib.load(p),
            "recommended_apply": bool(meta.get("recommended_apply", False)),
            "model_type": str(meta.get("model_type", "")),
        }
    return out


def predict_pwin_ml(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    df = df.copy()

    for c in ["signal_strength", "adx", "atrp", "base_weight", "p_win", "expected_return", "policy_score", "size_mult"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in ["symbol", "strategy_id", "side", "band", "reason"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    out = []
    for symbol, g in df.groupby("symbol", sort=False):
        g = g.copy()
        info = models.get(symbol)

        if info is None or not bool(info.get("recommended_apply", False)):
            g["pwin_ml"] = np.nan
            g["pwin_final"] = pd.to_numeric(g.get("p_win"), errors="coerce").fillna(0.5)
            g["pwin_source"] = "meta"
            out.append(g)
            continue

        bundle = info["bundle"]
        model = bundle["model"]
        num_cols = list(bundle.get("feature_numeric", []) or [])
        cat_cols = list(bundle.get("feature_categorical", []) or [])
        feat_cols = [c for c in (num_cols + cat_cols) if c in g.columns]

        X = g[feat_cols].copy()
        p = model.predict_proba(X)[:, 1]

        g["pwin_ml"] = p
        g["pwin_final"] = g["pwin_ml"]
        g["pwin_source"] = "ml"
        out.append(g)

    return pd.concat(out, axis=0, ignore_index=True)


def build_thresholds(df: pd.DataFrame) -> dict:
    asset_overrides = {}

    for symbol, g in df.groupby("symbol", sort=True):
        s = pd.to_numeric(g["pwin_final"], errors="coerce").dropna()
        if s.empty:
            continue

        q75 = float(s.quantile(0.75))
        q85 = float(s.quantile(0.85))
        q90 = float(s.quantile(0.90))
        q93 = float(s.quantile(0.93))

        # thresholds razonables ligados a la distribución real
        min_pwin_contextual = max(0.54, min(0.80, q75))
        min_pwin_strong = max(min_pwin_contextual + 0.015, min(0.88, q93))

        asset_overrides[symbol] = {
            "asset_gate": {
                "min_pwin_contextual": round(min_pwin_contextual, 6),
                "min_pwin_strong": round(min_pwin_strong, 6),
                "min_policy_score": 1e-06,
                "min_pwin_rank": 0.70,
                "min_postml_rank": 0.60,
                "min_competitive_rank": 0.55,
            },
            "diagnostics": {
                "pwin_min": round(float(s.min()), 6),
                "pwin_p50": round(float(s.quantile(0.50)), 6),
                "pwin_p75": round(q75, 6),
                "pwin_p85": round(q85, 6),
                "pwin_p90": round(q90, 6),
                "pwin_p93": round(q93, 6),
                "pwin_max": round(float(s.max()), 6),
                "rows": int(len(s)),
                "ml_share": round(float((g["pwin_source"] == "ml").mean()), 6),
            },
        }

    cfg = {
        "default_profile": "research",
        "profiles": {
            "research": {
                "asset_gate": {
                    "min_pwin_strong": 0.56,
                    "min_pwin_contextual": 0.54,
                    "min_policy_score": 1e-06,
                    "min_pwin_rank": 0.70,
                    "min_postml_rank": 0.60,
                    "min_competitive_rank": 0.55,
                },
                "asset_rank": {
                    "top_k_per_symbol": 1
                },
                "universe_gate": {
                    "min_global_score_rank": 0.65
                },
                "universe_rank": {
                    "top_frac": 0.35,
                    "min_keep": 1
                },
                "trace": {
                    "enabled": True
                }
            }
        },
        "asset_overrides": asset_overrides,
        "asset_side_overrides": {}
    }
    return cfg


def main() -> None:
    df = pd.read_csv(CANDIDATES_CSV)
    registry = load_registry(REGISTRY_JSON)
    models = load_models(registry)
    df = predict_pwin_ml(df, models)

    cfg = build_thresholds(df)
    Path(OUT_JSON).write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("saved:", OUT_JSON)
    print("\n=== THRESHOLD SUMMARY ===")
    rows = []
    for symbol, meta in (cfg.get("asset_overrides", {}) or {}).items():
        d = meta.get("diagnostics", {}) or {}
        ag = meta.get("asset_gate", {}) or {}
        rows.append({
            "symbol": symbol,
            "rows": d.get("rows"),
            "ml_share": d.get("ml_share"),
            "p50": d.get("pwin_p50"),
            "p75": d.get("pwin_p75"),
            "p90": d.get("pwin_p90"),
            "p93": d.get("pwin_p93"),
            "max": d.get("pwin_max"),
            "min_pwin_contextual": ag.get("min_pwin_contextual"),
            "min_pwin_strong": ag.get("min_pwin_strong"),
        })
    if rows:
        print(pd.DataFrame(rows).sort_values("symbol").to_string(index=False))
    else:
        print("(empty)")
        

if __name__ == "__main__":
    main()
