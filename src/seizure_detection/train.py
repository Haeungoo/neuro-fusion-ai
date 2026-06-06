from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
)

from src.common.paths import PROJECT_ROOT
from src.common.plotting import save_confusion_matrix


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    return float(tn / (tn + fp + 1e-12))


def train_random_forest_seizure(
    X_features: np.ndarray,
    y: np.ndarray,
    model_path: str | Path = "models/seizure_rf.pkl",
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    X_features = np.asarray(X_features, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    if X_features.ndim != 2:
        raise ValueError(f"X_features must be 2D. Got {X_features.shape}")

    if y.ndim != 1:
        raise ValueError(f"y must be 1D. Got {y.shape}")

    if len(X_features) != len(y):
        raise ValueError("X_features and y length mismatch.")

    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes: 0 = non-seizure, 1 = seizure.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "sensitivity": float(recall_score(y_test, y_pred)),
        "specificity": float(calculate_specificity(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
    }

    model_path = PROJECT_ROOT / model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": model,
        "metrics": metrics,
        "feature_count": int(X_features.shape[1]),
        "classes": ["non_seizure", "seizure"],
        "random_state": random_state,
    }

    joblib.dump(model_bundle, model_path)

    cm_path = PROJECT_ROOT / "results/seizure/confusion_matrix_seizure.png"
    save_confusion_matrix(
        y_test,
        y_pred,
        labels=["Non-seizure", "Seizure"],
        output_path=cm_path,
    )

    metrics_path = PROJECT_ROOT / "results/seizure/seizure_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Non-seizure", "Seizure"],
    )

    return {
        "model_path": str(model_path),
        "confusion_matrix_path": str(cm_path),
        "metrics_path": str(metrics_path),
        "classification_report": report,
        "example_probabilities": y_prob[:20].tolist(),
        **metrics,
    }