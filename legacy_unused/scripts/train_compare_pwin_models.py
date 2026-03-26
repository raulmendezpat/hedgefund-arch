from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _safe_read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)


def _to_utc_dt(series: pd.Series) -> pd.Series:
    s = series.copy()

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="ms", utc=True, errors="coerce")

    s_num = pd.to_numeric(s, errors="coerce")
    share_numeric = float(s_num.notna().mean()) if len(s_num) else 0.0

    if share_numeric >= 0.95:
        return pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce")

    return pd.to_datetime(s, utc=True, errors="coerce")


def _clip_prob(x: np.ndarray | pd.Series, lo: float = 1e-6, hi: float = 1 - 1e-6) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.clip(arr, lo, hi)


def _metrics(y_true: pd.Series, p: np.ndarray, label: str) -> dict:
    p = _clip_prob(p)
    y_hat = (p >= 0.5).astype(int)
    out = {
        "model": label,
        "n": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "prob_mean": float(np.mean(p)),
    }
    return out


def _quintile_report(df: pd.DataFrame, prob_col: str, y_col: str = "is_win", pnl_col: str = "pnl") -> pd.DataFrame:
    x = df[[prob_col, y_col, pnl_col]].copy()
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return pd.DataFrame()

    x["bin"] = pd.qcut(x[prob_col], q=min(5, x[prob_col].nunique()), duplicates="drop")
    g = x.groupby("bin", observed=False).agg(
        trades=(y_col, "size"),
        win_rate=(y_col, "mean"),
        pnl_sum=(pnl_col, "sum"),
        pnl_mean=(pnl_col, "mean"),
        prob_min=(prob_col, "min"),
        prob_max=(prob_col, "max"),
        prob_mean=(prob_col, "mean"),
    ).reset_index()
    g["model_prob"] = prob_col
    return g


def _build_dataset(candidates_path: Path, trades_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    cand = _safe_read_csv(candidates_path)
    trades = _safe_read_csv(trades_path)

    cand["ts_dt"] = _to_utc_dt(cand["ts"])
    trades["entry_dt"] = _to_utc_dt(trades["entry_ts"])

    join_cols_left = ["strategy_id", "symbol", "side", "ts_dt"]
    join_cols_right = ["strategy_id", "symbol", "side", "entry_dt"]

    merged = cand.merge(
        trades,
        left_on=join_cols_left,
        right_on=join_cols_right,
        how="inner",
        suffixes=("_cand", "_trade"),
    ).copy()

    if merged.empty:
        raise RuntimeError("Join candidates->trades returned 0 rows.")

    merged["is_win"] = (pd.to_numeric(merged["pnl"], errors="coerce").fillna(0.0) > 0.0).astype(int)

    numeric_features = [
        "signal_strength",
        "base_weight",
        "score",
        "policy_score",
        "expected_return",
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
        "portfolio_breadth",
        "portfolio_avg_pwin",
        "portfolio_avg_atrp",
        "portfolio_avg_strength",
        "portfolio_conviction",
    ]

    categorical_features = [
        "strategy_id",
        "symbol",
        "side",
        "band",
        "reason",
        "policy_profile",
        "portfolio_regime",
    ]

    for c in numeric_features:
        if c not in merged.columns:
            merged[c] = np.nan

    for c in categorical_features:
        if c not in merged.columns:
            merged[c] = ""

    keep_cols = (
        ["entry_dt", "pnl", "is_win", "p_win", "p_win_math_v1", "p_win_hybrid_v1"]
        + numeric_features
        + categorical_features
    )
    for col in ["p_win", "p_win_math_v1", "p_win_hybrid_v1"]:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged[keep_cols].copy()
    merged = merged.sort_values("entry_dt").reset_index(drop=True)

    return merged, numeric_features, categorical_features


def _time_split(df: pd.DataFrame, train_frac: float = 0.70, valid_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 30:
        raise RuntimeError(f"Too few rows for train/valid/test split: {n}")

    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + valid_frac))

    train = df.iloc[:i1].copy()
    valid = df.iloc[i1:i2].copy()
    test = df.iloc[i2:].copy()

    if train.empty or valid.empty or test.empty:
        raise RuntimeError("Invalid split produced empty partition.")

    return train, valid, test


def _make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def _fit_model(name: str, model, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    feats = num_cols + cat_cols

    pre = _make_preprocessor(num_cols, cat_cols)
    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", model),
        ]
    )

    X_train = train_df[feats]
    y_train = train_df["is_win"].astype(int)

    X_valid = valid_df[feats]
    y_valid = valid_df["is_win"].astype(int)

    X_test = test_df[feats]
    y_test = test_df["is_win"].astype(int)

    pipe.fit(X_train, y_train)

    p_valid = pipe.predict_proba(X_valid)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    metrics_rows = []
    metrics_rows.append({**_metrics(y_valid, p_valid, f"{name}__valid"), "split": "valid"})
    metrics_rows.append({**_metrics(y_test, p_test, f"{name}__test"), "split": "test"})

    pred_test = test_df[["entry_dt", "pnl", "is_win"]].copy()
    pred_test[f"p_{name}"] = p_test

    return pd.DataFrame(metrics_rows), pred_test, pipe


def _feature_importance_logistic(pipe: Pipeline, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    try:
        feat_names = list(pre.get_feature_names_out())
        coef = clf.coef_[0]
        out = pd.DataFrame({
            "feature": feat_names,
            "importance_abs": np.abs(coef),
            "importance_signed": coef,
            "model": "logistic",
        }).sort_values("importance_abs", ascending=False)
        return out
    except Exception:
        return pd.DataFrame(columns=["feature", "importance_abs", "importance_signed", "model"])


def _feature_importance_permutation(pipe: Pipeline, test_df: pd.DataFrame, num_cols: list[str], cat_cols: list[str], model_name: str) -> pd.DataFrame:
    feats = num_cols + cat_cols
    X_test = test_df[feats]
    y_test = test_df["is_win"].astype(int)
    try:
        r = permutation_importance(
            pipe,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="roc_auc",
        )
        out = pd.DataFrame({
            "feature": feats,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
            "model": model_name,
        }).sort_values("importance_mean", ascending=False)
        return out
    except Exception:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std", "model"])


def _baseline_eval(valid_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    pred_frames = []

    for split_name, df in [("valid", valid_df), ("test", test_df)]:
        y = df["is_win"].astype(int)

        for col in ["p_win", "p_win_math_v1", "p_win_hybrid_v1"]:
            if col in df.columns and df[col].notna().any():
                p = _clip_prob(df[col].fillna(0.5).to_numpy())
                metrics_rows.append({**_metrics(y, p, f"{col}__{split_name}"), "split": split_name})
                if split_name == "test":
                    tmp = df[["entry_dt", "pnl", "is_win"]].copy()
                    tmp[col] = p
                    pred_frames.append(tmp)

    pred_merged = pred_frames[0].copy() if pred_frames else pd.DataFrame()
    for extra in pred_frames[1:]:
        pred_merged = pred_merged.merge(extra, on=["entry_dt", "pnl", "is_win"], how="outer")

    return pd.DataFrame(metrics_rows), pred_merged


def _topk_numeric_features(importance_df: pd.DataFrame, num_cols: list[str], topk: int = 12) -> list[str]:
    if importance_df.empty:
        return num_cols[:topk]
    importance_df = importance_df.copy()
    importance_df["base_feature"] = importance_df["feature"].astype(str).str.replace(r"^num__", "", regex=True)
    ranked = []
    seen = set()
    for f in importance_df["base_feature"].tolist():
        if f in num_cols and f not in seen:
            ranked.append(f)
            seen.add(f)
        if len(ranked) >= topk:
            break
    return ranked if ranked else num_cols[:topk]


def _blend_search(valid_df: pd.DataFrame, test_df: pd.DataFrame, *, ml_col: str, math_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = valid_df.copy()
    test = test_df.copy()

    target_col = "target" if "target" in valid.columns else "is_win"
    if target_col not in valid.columns:
        raise ValueError("valid_df no contiene target ni is_win")

    for df in (valid, test):
        if ml_col not in df.columns:
            raise ValueError(f"Falta columna ML en blend_search: {ml_col}")
        if math_col not in df.columns:
            raise ValueError(f"Falta columna matemática en blend_search: {math_col}")

        df[ml_col] = pd.to_numeric(df[ml_col], errors="coerce")
        df[math_col] = pd.to_numeric(df[math_col], errors="coerce")

    valid[target_col] = pd.to_numeric(valid[target_col], errors="coerce")
    valid = valid.dropna(subset=[target_col, ml_col, math_col]).copy()

    if valid.empty:
        raise ValueError("No hay filas válidas para blend_search después de filtrar NaN.")

    valid[target_col] = valid[target_col].astype(int)

    test[ml_col] = pd.to_numeric(test[ml_col], errors="coerce").fillna(0.5)
    test[math_col] = pd.to_numeric(test[math_col], errors="coerce").fillna(0.5)

    grid_rows = []
    best_auc = -1.0
    best_acc = -1.0
    best_pred = None
    best_w_ml = None
    best_w_math = None

    y_valid = valid[target_col].to_numpy(dtype=int)

    for w_ml in np.linspace(0.0, 1.0, 21):
        w_math = 1.0 - float(w_ml)

        p_valid = (
            w_ml * valid[ml_col].to_numpy(dtype=float)
            + w_math * valid[math_col].to_numpy(dtype=float)
        )
        p_test = (
            w_ml * test[ml_col].to_numpy(dtype=float)
            + w_math * test[math_col].to_numpy(dtype=float)
        )

        p_valid = np.clip(p_valid, 0.0, 1.0)
        p_test = np.clip(p_test, 0.0, 1.0)

        auc = roc_auc_score(y_valid, p_valid) if len(np.unique(y_valid)) > 1 else np.nan
        acc = accuracy_score(y_valid, (p_valid >= 0.5).astype(int))

        grid_rows.append(
            {
                "ml_col": str(ml_col),
                "math_col": str(math_col),
                "blend_col": f"blend_{ml_col}_vs_{math_col}",
                "w_ml": float(w_ml),
                "w_math": float(w_math),
                "auc": float(auc) if pd.notna(auc) else np.nan,
                "accuracy": float(acc),
            }
        )

        auc_cmp = float(auc) if pd.notna(auc) else -1.0
        if (auc_cmp > best_auc) or (auc_cmp == best_auc and acc > best_acc):
            best_auc = auc_cmp
            best_acc = float(acc)
            best_pred = p_test.copy()
            best_w_ml = float(w_ml)
            best_w_math = float(w_math)

    if best_pred is None:
        raise ValueError("No se pudo construir blend prediction")

    grid_df = (
        pd.DataFrame(grid_rows)
        .sort_values(["auc", "accuracy"], ascending=[False, False])
        .reset_index(drop=True)
    )

    blend_col = f"blend_{ml_col}_vs_{math_col}"
    test_out = test[["entry_dt", "pnl", "is_win"]].copy()
    test_out[blend_col] = np.clip(best_pred, 0.0, 1.0)
    test_out["blend_w_ml"] = float(best_w_ml)
    test_out["blend_w_math"] = float(best_w_math)

    return grid_df, test_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out-prefix", default="results/pwin_ml_compare")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df, num_cols, cat_cols = _build_dataset(
        candidates_path=Path(args.candidates),
        trades_path=Path(args.trades),
    )

    train_df, valid_df, test_df = _time_split(df)

    baseline_metrics, baseline_test_pred = _baseline_eval(valid_df, test_df)

    logistic = LogisticRegression(
        max_iter=2000,
        C=0.5,
        class_weight="balanced",
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    hgb = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.03,
        max_iter=250,
        min_samples_leaf=20,
        random_state=42,
    )

    all_metrics = []
    all_test_preds = []

    m1, p1, pipe_log = _fit_model("logistic_full", logistic, train_df, valid_df, test_df, num_cols, cat_cols)
    m2, p2, pipe_rf = _fit_model("rf_full", rf, train_df, valid_df, test_df, num_cols, cat_cols)
    m3, p3, pipe_hgb = _fit_model("hgb_full", hgb, train_df, valid_df, test_df, num_cols, cat_cols)

    all_metrics.extend([baseline_metrics, m1, m2, m3])

    test_pred = test_df[["entry_dt", "pnl", "is_win", "p_win_math_v1"]].copy()
    if "p_win" in baseline_test_pred.columns:
        test_pred["p_win"] = baseline_test_pred["p_win"]
    if "p_win_hybrid_v1" in baseline_test_pred.columns:
        test_pred["p_win_hybrid_v1"] = baseline_test_pred["p_win_hybrid_v1"]

    test_pred = test_pred.merge(p1, on=["entry_dt", "pnl", "is_win"], how="left")
    test_pred = test_pred.merge(p2, on=["entry_dt", "pnl", "is_win"], how="left")
    test_pred = test_pred.merge(p3, on=["entry_dt", "pnl", "is_win"], how="left")

    valid_pred = valid_df[["entry_dt", "pnl", "is_win", "p_win_math_v1"]].copy()
    valid_pred["target"] = valid_pred["is_win"].astype(int)
    feats = num_cols + cat_cols
    valid_pred["p_logistic_full"] = pipe_log.predict_proba(valid_df[feats])[:, 1]
    valid_pred["p_rf_full"] = pipe_rf.predict_proba(valid_df[feats])[:, 1]
    valid_pred["p_hgb_full"] = pipe_hgb.predict_proba(valid_df[feats])[:, 1]

    # Feature reduction from logistic importance
    imp_log = _feature_importance_logistic(pipe_log, num_cols, cat_cols)
    perm_log = _feature_importance_permutation(pipe_log, test_df, num_cols, cat_cols, "logistic_full")
    perm_rf = _feature_importance_permutation(pipe_rf, test_df, num_cols, cat_cols, "rf_full")
    perm_hgb = _feature_importance_permutation(pipe_hgb, test_df, num_cols, cat_cols, "hgb_full")

    top_num = _topk_numeric_features(perm_log if not perm_log.empty else imp_log, num_cols, topk=12)

    m4, p4, pipe_log_top = _fit_model("logistic_top12", logistic, train_df, valid_df, test_df, top_num, cat_cols)
    m5, p5, pipe_hgb_top = _fit_model("hgb_top12", hgb, train_df, valid_df, test_df, top_num, cat_cols)

    all_metrics.extend([m4, m5])
    test_pred = test_pred.merge(p4, on=["entry_dt", "pnl", "is_win"], how="left")
    test_pred = test_pred.merge(p5, on=["entry_dt", "pnl", "is_win"], how="left")

    valid_pred["p_logistic_top12"] = pipe_log_top.predict_proba(valid_df[top_num + cat_cols])[:, 1]
    valid_pred["p_hgb_top12"] = pipe_hgb_top.predict_proba(valid_df[top_num + cat_cols])[:, 1]

    # Blend search
    blend_reports = []
    blend_preds = []

    for ml_col in ["p_logistic_full", "p_rf_full", "p_hgb_full", "p_logistic_top12", "p_hgb_top12"]:
        grid, pred = _blend_search(valid_pred, test_pred, ml_col=ml_col, math_col="p_win_math_v1")
        if not grid.empty:
            blend_reports.append(grid)
        if not pred.empty:
            test_pred = test_pred.merge(pred, on=["entry_dt", "pnl", "is_win"], how="left")
            blend_col = [c for c in pred.columns if c.startswith("blend_")][0]
            all_metrics.append(pd.DataFrame([{**_metrics(test_pred["is_win"], test_pred[blend_col], f"{blend_col}__test"), "split": "test"}]))
            all_test_preds.append(pred)

    metrics_df = pd.concat(all_metrics, ignore_index=True).sort_values(["split", "auc", "pr_auc"], ascending=[True, False, False])

    quintile_frames = []
    for col in [
        "p_win",
        "p_win_math_v1",
        "p_win_hybrid_v1",
        "p_logistic_full",
        "p_rf_full",
        "p_hgb_full",
        "p_logistic_top12",
        "p_hgb_top12",
    ] + [c for c in test_pred.columns if c.startswith("blend_")]:
        if col in test_pred.columns:
            q = _quintile_report(test_pred, col)
            if not q.empty:
                quintile_frames.append(q)

    quintiles_df = pd.concat(quintile_frames, ignore_index=True) if quintile_frames else pd.DataFrame()

    feature_imp_df = pd.concat(
        [
            imp_log.assign(kind="coef"),
            perm_log.assign(kind="permutation"),
            perm_rf.assign(kind="permutation"),
            perm_hgb.assign(kind="permutation"),
        ],
        ignore_index=True,
        sort=False,
    )

    # Calibration on test
    calib_rows = []
    for col in [
        "p_win",
        "p_win_math_v1",
        "p_win_hybrid_v1",
        "p_logistic_full",
        "p_rf_full",
        "p_hgb_full",
        "p_logistic_top12",
        "p_hgb_top12",
    ] + [c for c in test_pred.columns if c.startswith("blend_")]:
        if col not in test_pred.columns:
            continue
        yt = test_pred["is_win"].astype(int).to_numpy()
        pt = _clip_prob(test_pred[col].fillna(0.5).to_numpy())
        try:
            prob_true, prob_pred = calibration_curve(yt, pt, n_bins=8, strategy="quantile")
            for a, b in zip(prob_pred, prob_true):
                calib_rows.append({
                    "model_prob": col,
                    "pred_bin_mean": float(a),
                    "true_rate": float(b),
                })
        except Exception:
            pass

    calibration_df = pd.DataFrame(calib_rows)

    comparison_path = Path(f"{out_prefix}_comparison.csv")
    quintiles_path = Path(f"{out_prefix}_quintiles.csv")
    features_path = Path(f"{out_prefix}_feature_importance.csv")
    preds_path = Path(f"{out_prefix}_test_predictions.csv")
    calibration_path = Path(f"{out_prefix}_calibration.csv")
    blend_grid_path = Path(f"{out_prefix}_blend_grid.csv")
    summary_json_path = Path(f"{out_prefix}_summary.json")

    metrics_df.to_csv(comparison_path, index=False)
    quintiles_df.to_csv(quintiles_path, index=False)
    feature_imp_df.to_csv(features_path, index=False)
    test_pred.to_csv(preds_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)

    if blend_reports:
        pd.concat(blend_reports, ignore_index=True).to_csv(blend_grid_path, index=False)
    else:
        pd.DataFrame().to_csv(blend_grid_path, index=False)

    summary = {
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "top_numeric_features": list(top_num),
        "best_test_models_by_auc": metrics_df[metrics_df["split"].eq("test")].head(10).to_dict(orient="records"),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== DATASET ===")
    print({
        "total_rows": len(df),
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
        "positive_rate_total": float(df["is_win"].mean()),
        "positive_rate_test": float(test_df["is_win"].mean()),
    })

    print("\n=== TOP TEST MODELS BY AUC ===")
    print(
        metrics_df[metrics_df["split"].eq("test")]
        .sort_values(["auc", "pr_auc", "accuracy"], ascending=False)
        .head(15)
        .to_string(index=False)
    )

    print("\n=== TOP NUMERIC FEATURES ===")
    print(top_num)

    print("\n=== FILES SAVED ===")
    print(comparison_path)
    print(quintiles_path)
    print(features_path)
    print(preds_path)
    print(calibration_path)
    print(blend_grid_path)
    print(summary_json_path)


if __name__ == "__main__":
    main()
