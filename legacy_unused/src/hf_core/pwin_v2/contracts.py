from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DatasetBundle:
    df: Any
    numeric_features: list[str]
    categorical_features: list[str]
    key_cols: list[str]


@dataclass(slots=True)
class SplitBundle:
    train_df: Any
    valid_df: Any
    test_df: Any


@dataclass(slots=True)
class ModelMetrics:
    model_name: str
    scope: str
    train_rows: int
    valid_rows: int
    test_rows: int
    positive_rate_train: float
    positive_rate_valid: float
    positive_rate_test: float
    accuracy_valid: float
    balanced_accuracy_valid: float
    roc_auc_valid: float | None
    brier_valid: float | None
    accuracy_test: float
    balanced_accuracy_test: float
    roc_auc_test: float | None
    brier_test: float | None
    recommended_apply: bool
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainedArtifact:
    model: Any
    model_name: str
    scope: str
    feature_numeric: list[str]
    feature_categorical: list[str]
    key_value: str
    metrics: dict[str, Any]
    training_meta: dict[str, Any] = field(default_factory=dict)
