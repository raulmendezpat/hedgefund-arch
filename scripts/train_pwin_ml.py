from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from hf_core.ml.sklearn_utils import FunctionTransformerDense

DEFAULT_NUMERIC = [
    "signal_strength",
    "adx",
    "atrp",
    "base_weight",
    "p_win",
    "expected_return",
    "policy_score",
    "size_mult",
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

DEFAULT_CATEGORICAL = [
    "symbol",
    "strategy_id",
    "side",
    "band",
    "reason",
]

TARGET_HORIZON_TO_RET = {
    "1h": "ret_1h",
    "4h": "ret_4h",
    "12h": "ret_12h",
    "24h": "ret_24h",
    "48h": "ret_48h",
}


def _safe_auc(y_true, y_prob):
    try:
        y_true = pd.Series(y_true).astype(int)
        if y_true.nunique() < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = pd.to_numeric(pd.Series(y_true), errors="coerce").fillna(0).astype(int)
    y_prob = pd.to_numeric(pd.Series(y_prob), errors="coerce").fillna(0.5).astype(float)
    y_hat = (y_prob >= float(threshold)).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
        "roc_auc": _safe_auc(y_true, y_prob),
    }
    try:
        out["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        out["brier"] = None
    return out


def recommended_apply(metrics, min_accuracy, min_bal_acc, min_auc):
    auc = metrics.get("roc_auc", None)
    return (
        float(metrics.get("accuracy", 0.0) or 0.0) >= float(min_accuracy)
        and float(metrics.get("balanced_accuracy", 0.0) or 0.0) >= float(min_bal_acc)
        and auc is not None
        and float(auc) >= float(min_auc)
    )


def resolve_feature_cols(df, numeric_cols, categorical_cols):
    nums = [c for c in numeric_cols if c in df.columns]
    cats = [c for c in categorical_cols if c in df.columns]
    return nums, cats


def split_timewise(df, test_frac=0.25):
    df = df.sort_values("ts").reset_index(drop=True).copy()
    cut = max(1, int(len(df) * (1.0 - test_frac)))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def prepare_dataset(candidates_df: pd.DataFrame, attribution_df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    ret_col = TARGET_HORIZON_TO_RET[horizon]

    left = candidates_df.copy()
    right = attribution_df[["ts", "symbol", "strategy_id", "side", ret_col]].copy()

    for df in [left, right]:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df["symbol"] = df["symbol"].astype(str).str.strip()
        df["strategy_id"] = df["strategy_id"].astype(str).str.strip()
        df["side"] = df["side"].astype(str).str.strip().str.lower()

    grp = ["symbol", "strategy_id", "side"]

    left = left.sort_values(grp + ["ts"]).reset_index(drop=True)
    right = right.sort_values(grp + ["ts"]).reset_index(drop=True)

    left["_seq"] = left.groupby(grp).cumcount()
    right["_seq"] = right.groupby(grp).cumcount()

    df = left.merge(
        right,
        on=["symbol", "strategy_id", "side", "_seq"],
        how="inner",
        suffixes=("", "_attr"),
    ).copy()

    df["label_win"] = (pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0) > 0.0).astype(int)
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def make_preprocessor(nums, cats):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, nums),
            ("cat", cat_pipe, cats),
        ],
        remainder="drop",
    )


def build_models(nums, cats):
    pre = make_preprocessor(nums, cats)

    logreg = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    mlp = Pipeline([
        ("pre", pre),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            random_state=42,
        )),
    ])

    # HGB necesita matriz densa numérica -> usamos OHE + scaler del pre y luego convertimos
    hgb = Pipeline([
        ("pre", pre),
        ("to_dense", FunctionTransformerDense()),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=250,
            min_samples_leaf=40,
            l2_regularization=0.1,
            random_state=42,
        )),
    ])

    extratrees = Pipeline([
        ("pre", pre),
        ("to_dense", FunctionTransformerDense()),
        ("clf", ExtraTreesClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    return {
        "logreg": logreg,
        "mlp": mlp,
        "hgb": hgb,
        "extratrees": extratrees,
    }




def train_best_model(
    df_all: pd.DataFrame,
    target_symbol: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    min_rows_symbol: int,
    test_frac: float,
    min_accuracy: float,
    min_bal_acc: float,
    min_auc: float,
):
    df_target = df_all[df_all["symbol"].eq(target_symbol)].copy()
    fallback_mode = "symbol"

    if len(df_target) >= int(min_rows_symbol):
        train_df, test_df = split_timewise(df_target, test_frac=test_frac)
        train_source = train_df.copy()
        test_source = test_df.copy()
    else:
        fallback_mode = "round_robin_ex_symbol"
        df_pool = df_all[~df_all["symbol"].eq(target_symbol)].copy()
        train_df, test_df = split_timewise(df_target, test_frac=test_frac)
        train_source = df_pool.copy()
        test_source = test_df.copy()

    if train_source.empty or test_source.empty:
        raise ValueError(f"Insufficient train/test rows for target_symbol={target_symbol}")

    nums, cats = resolve_feature_cols(df_all, numeric_cols, categorical_cols)
    feat_cols = nums + cats

    X_train = train_source[feat_cols].copy()
    y_train = train_source["label_win"].copy()
    X_test = test_source[feat_cols].copy()
    y_test = test_source["label_win"].copy()

    models = build_models(nums, cats)

    best_name = None
    best_model = None
    best_metrics = None

    all_results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob)
        all_results[model_name] = metrics

        score = (
            (metrics["roc_auc"] if metrics["roc_auc"] is not None else -1.0),
            metrics["balanced_accuracy"],
            metrics["accuracy"],
        )

        if best_metrics is None:
            best_name, best_model, best_metrics = model_name, model, metrics
        else:
            prev = (
                (best_metrics["roc_auc"] if best_metrics["roc_auc"] is not None else -1.0),
                best_metrics["balanced_accuracy"],
                best_metrics["accuracy"],
            )
            if score > prev:
                best_name, best_model, best_metrics = model_name, model, metrics

    result = {
        "target_key": target_symbol,
        "train_rows": int(len(train_source)),
        "test_rows": int(len(test_source)),
        "model_type": str(best_name),
        "accuracy": float(best_metrics["accuracy"]),
        "balanced_accuracy": float(best_metrics["balanced_accuracy"]),
        "roc_auc": None if best_metrics["roc_auc"] is None else float(best_metrics["roc_auc"]),
        "brier": None if best_metrics["brier"] is None else float(best_metrics["brier"]),
        "positive_rate_train": float(pd.to_numeric(y_train, errors="coerce").fillna(0).mean()),
        "positive_rate_test": float(pd.to_numeric(y_test, errors="coerce").fillna(0).mean()),
        "recommended_apply": bool(recommended_apply(best_metrics, min_accuracy, min_bal_acc, min_auc)),
        "fallback_mode": fallback_mode,
        "feature_numeric": nums,
        "feature_categorical": cats,
        "all_results": all_results,
    }

    bundle = {
        "model": best_model,
        "model_type": best_name,
        "target_symbol": target_symbol,
        "fallback_mode": fallback_mode,
        "feature_numeric": nums,
        "feature_categorical": cats,
        "metrics": result,
    }
    return bundle, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-csv", required=True)
    ap.add_argument("--attribution-csv", required=True)
    ap.add_argument("--horizon", default="24h", choices=["1h", "4h", "12h", "24h", "48h"])
    ap.add_argument("--out-dir", default="artifacts/pwin_ml")
    ap.add_argument("--min-rows-symbol", type=int, default=1500)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--min-accuracy", type=float, default=0.54)
    ap.add_argument("--min-balanced-accuracy", type=float, default=0.53)
    ap.add_argument("--min-auc", type=float, default=0.56)
    args = ap.parse_args()

    candidates_df = pd.read_csv(args.candidates_csv)
    attribution_df = pd.read_csv(args.attribution_csv)

    print("candidates_rows_raw:", len(candidates_df))
    print("attribution_rows_raw:", len(attribution_df))

    df_all = prepare_dataset(candidates_df, attribution_df, args.horizon)
    print("prepared_dataset_rows_after_merge:", len(df_all))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    summary_rows = []
    registry = {
        "horizon": str(args.horizon),
        "candidates_csv": str(args.candidates_csv),
        "attribution_csv": str(args.attribution_csv),
        "models": {},
    }

    symbols = sorted(df_all["symbol"].astype(str).unique().tolist())
    print("prepared_symbols:", symbols)

    for symbol in symbols:
        try:
            bundle, result = train_best_model(
                df_all=df_all,
                target_symbol=symbol,
                numeric_cols=list(DEFAULT_NUMERIC),
                categorical_cols=list(DEFAULT_CATEGORICAL),
                min_rows_symbol=int(args.min_rows_symbol),
                test_frac=float(args.test_frac),
                min_accuracy=float(args.min_accuracy),
                min_bal_acc=float(args.min_balanced_accuracy),
                min_auc=float(args.min_auc),
            )

            safe_symbol = symbol.replace("/", "_").replace(":", "_")
            model_path = str(Path(args.out_dir) / f"pwin_model_{safe_symbol}.joblib")
            joblib.dump(bundle, model_path)

            registry["models"][symbol] = {
                "path": model_path,
                "recommended_apply": bool(result["recommended_apply"]),
                "fallback_mode": str(result["fallback_mode"]),
                "model_type": str(result["model_type"]),
                "accuracy": float(result["accuracy"]),
                "balanced_accuracy": float(result["balanced_accuracy"]),
                "roc_auc": None if result["roc_auc"] is None else float(result["roc_auc"]),
                "brier": None if result["brier"] is None else float(result["brier"]),
                "all_results": result["all_results"],
            }
            summary_rows.append(result)
            print("trained:", symbol, "best=", result["model_type"], "auc=", result["roc_auc"])
        except Exception as e:
            print("SKIP:", symbol, "reason:", repr(e))

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        print("WARNING: summary_rows vacío; no se entrenó ningún modelo")
    else:
        for col in ["recommended_apply", "roc_auc", "balanced_accuracy", "accuracy"]:
            if col not in summary_df.columns:
                summary_df[col] = None
        summary_df = summary_df.sort_values(
            ["recommended_apply", "roc_auc", "balanced_accuracy", "accuracy"],
            ascending=[False, False, False, False],
            na_position="last",
        )

    summary_csv = str(Path(args.out_dir) / "pwin_ml_summary.csv")
    registry_json = str(Path(args.out_dir) / "pwin_ml_registry.json")

    summary_df.to_csv(summary_csv, index=False)
    Path(registry_json).write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print("saved:", summary_csv)
    print("saved:", registry_json)
    print("\n=== SUMMARY ===")
    if summary_df.empty:
        print("(empty)")
    else:
        cols = [
            "target_key", "train_rows", "test_rows", "model_type",
            "accuracy", "balanced_accuracy", "roc_auc", "brier",
            "recommended_apply", "fallback_mode"
        ]
        cols = [c for c in cols if c in summary_df.columns]
        print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    import json
    main()
