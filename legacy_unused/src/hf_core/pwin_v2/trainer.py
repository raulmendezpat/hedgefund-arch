from __future__ import annotations

from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .contracts import ModelMetrics, TrainedArtifact
from .dataset import time_split


def _safe_auc(y_true, p_pred) -> float | None:
    y = pd.Series(y_true).astype(int)
    if y.nunique() < 2:
        return None
    try:
        return float(roc_auc_score(y, p_pred))
    except Exception:
        return None


def _safe_brier(y_true, p_pred) -> float | None:
    try:
        return float(brier_score_loss(pd.Series(y_true).astype(int), pd.Series(p_pred).astype(float)))
    except Exception:
        return None


def _clip_prob(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.nan_to_num(arr, nan=0.5, posinf=0.999999, neginf=1e-6)
    return np.clip(arr, 1e-6, 1.0 - 1e-6)


def _metrics(y_true, p_pred) -> dict:
    y = pd.Series(y_true).astype(int)
    p = _clip_prob(p_pred)
    y_hat = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, y_hat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_hat)),
        "roc_auc": _safe_auc(y, p),
        "brier": _safe_brier(y, p),
    }


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
        ],
        remainder="drop",
    )


def _build_models(num_cols: list[str], cat_cols: list[str]) -> dict[str, Pipeline]:
    pre = _make_preprocessor(num_cols, cat_cols)

    return {
        "logistic": Pipeline(
            steps=[
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("pre", pre),
                ("clf", RandomForestClassifier(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_leaf=8,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                )),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("pre", pre),
                ("clf", ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=10,
                    min_samples_leaf=6,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                )),
            ]
        ),
        "hgb": Pipeline(
            steps=[
                ("pre", pre),
                ("clf", HistGradientBoostingClassifier(
                    max_depth=5,
                    learning_rate=0.03,
                    max_iter=300,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=42,
                )),
            ]
        ),
        "mlp_small": Pipeline(
            steps=[
                ("pre", pre),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    random_state=42,
                )),
            ]
        ),
        "mlp_medium": Pipeline(
            steps=[
                ("pre", pre),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=3e-4,
                    learning_rate_init=7e-4,
                    max_iter=700,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=42,
                )),
            ]
        ),
    }


def _fit_one_model(
    model_name: str,
    model: Pipeline,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[dict, np.ndarray, np.ndarray, Pipeline]:
    feat_cols = list(num_cols) + list(cat_cols)

    X_train = train_df[feat_cols].copy()
    y_train = train_df["is_win"].astype(int).copy()

    X_valid = valid_df[feat_cols].copy()
    y_valid = valid_df["is_win"].astype(int).copy()

    X_test = test_df[feat_cols].copy()
    y_test = test_df["is_win"].astype(int).copy()

    model.fit(X_train, y_train)

    p_valid = _clip_prob(model.predict_proba(X_valid)[:, 1])
    p_test = _clip_prob(model.predict_proba(X_test)[:, 1])

    mv = _metrics(y_valid, p_valid)
    mt = _metrics(y_test, p_test)

    out = {
        "model_name": model_name,
        "accuracy_valid": mv["accuracy"],
        "balanced_accuracy_valid": mv["balanced_accuracy"],
        "roc_auc_valid": mv["roc_auc"],
        "brier_valid": mv["brier"],
        "accuracy_test": mt["accuracy"],
        "balanced_accuracy_test": mt["balanced_accuracy"],
        "roc_auc_test": mt["roc_auc"],
        "brier_test": mt["brier"],
    }
    return out, p_valid, p_test, model


def _score_tuple(d: dict) -> tuple:
    auc = d.get("roc_auc_valid")
    return (
        -1.0 if auc is None else float(auc),
        float(d.get("balanced_accuracy_valid", 0.0) or 0.0),
        float(d.get("accuracy_valid", 0.0) or 0.0),
    )


def train_scope_model(
    df_scope: pd.DataFrame,
    *,
    scope_name: str,
    key_value: str,
    num_cols: list[str],
    cat_cols: list[str],
    min_rows: int = 120,
    min_auc: float = 0.54,
    min_bal_acc: float = 0.52,
    min_acc: float = 0.52,
) -> tuple[TrainedArtifact, pd.DataFrame, pd.DataFrame]:
    if len(df_scope) < int(min_rows):
        raise ValueError(f"{scope_name}::{key_value} sin suficientes filas: {len(df_scope)} < {min_rows}")

    split = time_split(df_scope)

    models = _build_models(num_cols, cat_cols)

    rows = []
    best_name = None
    best_pipe = None
    best_valid = None
    best_test = None
    best_result = None

    for model_name, model in models.items():
        result, p_valid, p_test, pipe = _fit_one_model(
            model_name=model_name,
            model=model,
            train_df=split.train_df,
            valid_df=split.valid_df,
            test_df=split.test_df,
            num_cols=num_cols,
            cat_cols=cat_cols,
        )
        rows.append(result)

        if best_result is None or _score_tuple(result) > _score_tuple(best_result):
            best_result = dict(result)
            best_name = model_name
            best_pipe = pipe
            best_valid = p_valid
            best_test = p_test

    assert best_result is not None
    assert best_pipe is not None
    assert best_valid is not None
    assert best_test is not None
    assert best_name is not None

    recommended_apply = (
        (best_result["roc_auc_valid"] is not None and float(best_result["roc_auc_valid"]) >= float(min_auc))
        and float(best_result["balanced_accuracy_valid"]) >= float(min_bal_acc)
        and float(best_result["accuracy_valid"]) >= float(min_acc)
    )

    metrics = ModelMetrics(
        model_name=str(best_name),
        scope=str(scope_name),
        train_rows=int(len(split.train_df)),
        valid_rows=int(len(split.valid_df)),
        test_rows=int(len(split.test_df)),
        positive_rate_train=float(split.train_df["is_win"].mean()),
        positive_rate_valid=float(split.valid_df["is_win"].mean()),
        positive_rate_test=float(split.test_df["is_win"].mean()),
        accuracy_valid=float(best_result["accuracy_valid"]),
        balanced_accuracy_valid=float(best_result["balanced_accuracy_valid"]),
        roc_auc_valid=None if best_result["roc_auc_valid"] is None else float(best_result["roc_auc_valid"]),
        brier_valid=None if best_result["brier_valid"] is None else float(best_result["brier_valid"]),
        accuracy_test=float(best_result["accuracy_test"]),
        balanced_accuracy_test=float(best_result["balanced_accuracy_test"]),
        roc_auc_test=None if best_result["roc_auc_test"] is None else float(best_result["roc_auc_test"]),
        brier_test=None if best_result["brier_test"] is None else float(best_result["brier_test"]),
        recommended_apply=bool(recommended_apply),
        meta={},
    )

    artifact = TrainedArtifact(
        model=best_pipe,
        model_name=str(best_name),
        scope=str(scope_name),
        feature_numeric=list(num_cols),
        feature_categorical=list(cat_cols),
        key_value=str(key_value),
        metrics=asdict(metrics),
        training_meta={},
    )

    valid_pred = split.valid_df[["entry_dt", "pnl", "is_win", "symbol", "side", "strategy_id"]].copy()
    valid_pred["p_ml"] = best_valid

    test_pred = split.test_df[["entry_dt", "pnl", "is_win", "symbol", "side", "strategy_id"]].copy()
    test_pred["p_ml"] = best_test

    leaderboard = pd.DataFrame(rows).sort_values(
        ["roc_auc_valid", "balanced_accuracy_valid", "accuracy_valid"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return artifact, leaderboard, pd.concat(
        [
            valid_pred.assign(split="valid"),
            test_pred.assign(split="test"),
        ],
        ignore_index=True,
    )


def save_artifact(path: str, artifact: TrainedArtifact) -> None:
    joblib.dump(artifact, path)
