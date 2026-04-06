from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = Path("results/pwin_asset_side_dataset_v2_clean.csv")
OUT_DIR = Path("artifacts/pwin_asset_side_models_v2_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25
MIN_ROWS_PER_GROUP = 80
MIN_CLASS_ROWS = 12

GROUP_KEYS = ["symbol", "side"]
TARGET_COL = "label_win"

NUMERIC_FEATURES = [
    "signal_strength",
    "base_weight",
    "adx",
    "atrp",
    "rsi",
    "ret_1h_lag",
    "ret_4h_lag",
    "ret_12h_lag",
    "ret_24h_lag",
    "ema_gap_fast_slow",
    "dist_close_ema_fast",
    "dist_close_ema_slow",
    "range_pct",
    "rolling_vol_24h",
    "rolling_vol_72h",
    "atrp_zscore",
    "breakout_distance_up",
    "breakout_distance_down",
    "pullback_depth",
    "btc_ret_24h_lag",
    "btc_rolling_vol_24h",
    "btc_atrp",
    "btc_adx",
]

CATEGORICAL_FEATURES = [
    "strategy_id",
]


@dataclass
class ModelSpec:
    name: str
    estimator: Any


def sanitize_key(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace(":", "_")
        .replace("|", "_")
        .replace(" ", "_")
        .lower()
    )


def safe_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    try:
        if pd.Series(y_true).nunique() < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return num_cols, cat_cols


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    transformers = []

    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )

    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "logreg_l2_c1",
            LogisticRegression(
                max_iter=3000,
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "logreg_l2_c03",
            LogisticRegression(
                max_iter=3000,
                C=0.3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "rf_300_d6",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "rf_500_d8",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "hgb_200_d4",
            HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "hgb_300_d6",
            HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=6,
                learning_rate=0.03,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def choose_best(df: pd.DataFrame) -> pd.Series:
    ranked = df.copy()
    ranked["auc_rank"] = pd.to_numeric(ranked["auc"], errors="coerce").fillna(-1.0)
    ranked["acc_rank"] = pd.to_numeric(ranked["accuracy"], errors="coerce").fillna(-1.0)
    ranked = ranked.sort_values(
        ["auc_rank", "acc_rank", "test_rows"],
        ascending=[False, False, False],
    )
    return ranked.iloc[0]


def main() -> None:
    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, low_memory=False)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"Target column not found: {TARGET_COL}")

    print(f"Using dataset: {DATASET_PATH}")
    print("rows:", len(df))

    all_results = []
    best_rows = []
    registry = {
        "version": "pwin_asset_side_models_v2_clean",
        "dataset_path": str(DATASET_PATH),
        "target_col": TARGET_COL,
        "groups": {},
    }

    for (symbol, side), g in df.groupby(GROUP_KEYS, dropna=False):
        g = g.copy()
        g[TARGET_COL] = pd.to_numeric(g[TARGET_COL], errors="coerce").fillna(0).astype(int)

        if len(g) < MIN_ROWS_PER_GROUP:
            print(f"skip {symbol}|{side}: rows={len(g)} < {MIN_ROWS_PER_GROUP}")
            continue

        pos = int(g[TARGET_COL].sum())
        neg = int(len(g) - pos)
        if pos < MIN_CLASS_ROWS or neg < MIN_CLASS_ROWS:
            print(f"skip {symbol}|{side}: pos={pos}, neg={neg}")
            continue

        num_cols, cat_cols = build_feature_lists(g)
        feature_cols = num_cols + cat_cols
        if not feature_cols:
            print(f"skip {symbol}|{side}: no feature cols")
            continue

        X = g[feature_cols].copy()
        y = g[TARGET_COL].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        group_results = []
        fitted_models: dict[str, dict[str, Any]] = {}

        for spec in get_model_specs():
            pre = build_preprocessor(num_cols, cat_cols)
            pipe = Pipeline(
                [
                    ("pre", pre),
                    ("clf", clone(spec.estimator)),
                ]
            )

            try:
                pipe.fit(X_train, y_train)

                if hasattr(pipe, "predict_proba"):
                    prob = pipe.predict_proba(X_test)[:, 1]
                else:
                    raw = pipe.decision_function(X_test)
                    prob = 1.0 / (1.0 + np.exp(-raw))

                pred = (prob >= 0.5).astype(int)
                acc = float(accuracy_score(y_test, pred))
                auc = safe_auc(y_test, prob)

                row = {
                    "symbol": symbol,
                    "side": side,
                    "group_name": f"{symbol}|{side}",
                    "model_name": spec.name,
                    "train_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                    "positive_rate_train": float(y_train.mean()),
                    "positive_rate_test": float(y_test.mean()),
                    "accuracy": acc,
                    "auc": auc,
                    "features": "|".join(feature_cols),
                }
                group_results.append(row)

                fitted_models[spec.name] = {
                    "pipeline": pipe,
                    "feature_cols": feature_cols,
                    "numeric_cols": num_cols,
                    "categorical_cols": cat_cols,
                    "symbol": symbol,
                    "side": side,
                    "target_col": TARGET_COL,
                    "version": "pwin_asset_side_models_v2_clean",
                }

            except Exception as e:
                group_results.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "group_name": f"{symbol}|{side}",
                        "model_name": spec.name,
                        "train_rows": int(len(X_train)),
                        "test_rows": int(len(X_test)),
                        "positive_rate_train": float(y_train.mean()),
                        "positive_rate_test": float(y_test.mean()),
                        "accuracy": float("nan"),
                        "auc": float("nan"),
                        "features": "|".join(feature_cols),
                        "error": str(e),
                    }
                )

        group_df = pd.DataFrame(group_results)
        all_results.append(group_df)

        best = choose_best(group_df)
        best_rows.append(best.to_dict())

        best_model_name = str(best["model_name"])
        payload = fitted_models.get(best_model_name)
        if payload is None:
            print(f"skip save {symbol}|{side}: best model payload missing")
            continue

        model_path = OUT_DIR / f"pwin_model_{sanitize_key(symbol)}_{sanitize_key(side)}.pkl"
        with model_path.open("wb") as f:
            pickle.dump(payload, f)

        registry["groups"][f"{symbol}|{side}"] = {
            "symbol": symbol,
            "side": side,
            "model_name": best_model_name,
            "model_path": str(model_path),
            "accuracy": best.get("accuracy"),
            "auc": best.get("auc"),
            "train_rows": int(best.get("train_rows", 0)),
            "test_rows": int(best.get("test_rows", 0)),
            "features": str(best.get("features", "")).split("|"),
        }

        print(
            f"best {symbol}|{side}: {best_model_name} | "
            f"acc={best.get('accuracy')} | auc={best.get('auc')}"
        )

    if not all_results:
        raise SystemExit("No models trained")

    sweep_df = pd.concat(all_results, ignore_index=True)
    best_df = pd.DataFrame(best_rows)

    sweep_csv = OUT_DIR / "pwin_asset_side_model_sweep_results.csv"
    best_csv = OUT_DIR / "pwin_asset_side_best_models.csv"
    registry_json = OUT_DIR / "pwin_asset_side_model_registry.json"

    sweep_df.to_csv(sweep_csv, index=False)
    best_df.sort_values(["auc", "accuracy"], ascending=[False, False]).to_csv(best_csv, index=False)
    registry_json.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print()
    print("saved:", sweep_csv)
    print("saved:", best_csv)
    print("saved:", registry_json)


if __name__ == "__main__":
    main()
