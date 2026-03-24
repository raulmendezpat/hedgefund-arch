from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainResult:
    target_key: str
    train_rows: int
    test_rows: int
    model_type: str
    accuracy: float
    balanced_accuracy: float
    roc_auc: float | None
    brier: float | None
    positive_rate_train: float
    positive_rate_test: float
    recommended_apply: bool
    fallback_mode: str


def _safe_auc(y_true, y_prob) -> float | None:
    try:
        if len(set(pd.Series(y_true).dropna().astype(int).tolist())) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(numeric_cols)),
            ("cat", cat_pipe, list(categorical_cols)),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])


def split_timewise(df: pd.DataFrame, test_frac: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("ts").reset_index(drop=True).copy()
    n = len(df)
    cut = max(1, int(n * (1.0 - test_frac)))
    train_df = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()
    return train_df, test_df


def compute_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    y_true = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    y_hat = (pd.Series(y_prob).fillna(0.5) >= float(threshold)).astype(int)

    acc = float(accuracy_score(y_true, y_hat))
    bacc = float(balanced_accuracy_score(y_true, y_hat))
    auc = _safe_auc(y_true, y_prob)

    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except Exception:
        brier = None

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "roc_auc": auc,
        "brier": brier,
    }


def recommended_apply(metrics: dict[str, Any], min_accuracy: float, min_bal_acc: float, min_auc: float) -> bool:
    acc = float(metrics.get("accuracy", 0.0) or 0.0)
    bacc = float(metrics.get("balanced_accuracy", 0.0) or 0.0)
    auc = metrics.get("roc_auc", None)
    auc_ok = (auc is not None) and (float(auc) >= float(min_auc))
    return (acc >= float(min_accuracy)) and (bacc >= float(min_bal_acc)) and auc_ok


def save_bundle(path: str, bundle: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str) -> dict[str, Any]:
    return joblib.load(path)


def save_json(path: str, obj: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
