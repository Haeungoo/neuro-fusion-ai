from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.common.paths import PROJECT_ROOT
from src.common.plotting import save_confusion_matrix


def save_csp_patterns(
    csp: CSP,
    info,
    output_path: str | Path = "results/motor/csp_patterns.png",
    max_components: int = 4,
) -> str:
    """
    Save real CSP spatial pattern topomap.

    This uses the trained CSP object and MNE Epochs info.
    """
    output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_components = getattr(csp, "n_components", max_components)
    components = list(range(min(max_components, n_components)))

    try:
        fig = csp.plot_patterns(
            info,
            components=components,
            ch_type="eeg",
            show=False,
        )

        # Some MNE versions return a Figure, some may return a list-like object.
        if isinstance(fig, list):
            fig = fig[0]

        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        print("Warning: failed to save CSP patterns.")
        print(e)

    return str(output_path)


def train_csp_lda(
    X: np.ndarray,
    y: np.ndarray,
    model_path: str | Path = "models/motor_csp_lda.pkl",
    test_size: float = 0.25,
    random_state: int = 42,
    n_components: int = 6,
    info=None,
) -> dict:
    """
    Train CSP + LDA motor imagery classifier.

    Parameters
    ----------
    X:
        EEG epochs, shape (n_epochs, n_channels, n_times)

    y:
        Class labels, shape (n_epochs,)

    info:
        MNE Epochs info. Needed to generate real CSP topomap.
    """

    X = np.asarray(X)
    y = np.asarray(y, dtype=int)

    if X.ndim != 3:
        raise ValueError(
            f"X must be 3D: (n_epochs, n_channels, n_times). Got shape {X.shape}"
        )

    if y.ndim != 1:
        raise ValueError(
            f"y must be 1D: (n_epochs,). Got shape {y.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"X and y length mismatch: {len(X)} vs {len(y)}"
        )

    if len(np.unique(y)) < 2:
        raise ValueError("Need two classes for motor imagery training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    csp = CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False,
    )

    lda = LinearDiscriminantAnalysis()

    X_train_csp = csp.fit_transform(X_train, y_train)
    X_test_csp = csp.transform(X_test)

    lda.fit(X_train_csp, y_train)

    y_pred = lda.predict(X_test_csp)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "n_epochs": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "test_size": float(test_size),
        "random_state": int(random_state),
    }

    if info is not None:
        metrics["sfreq"] = float(info["sfreq"])
        metrics["ch_names"] = list(info["ch_names"])

    model_path = PROJECT_ROOT / model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "csp": csp,
        "lda": lda,
        "metrics": metrics,
        "classes": ["class_0", "class_1"],
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "random_state": int(random_state),
    }

    joblib.dump(model_bundle, model_path)

    cm_path = PROJECT_ROOT / "results/motor/confusion_matrix_motor.png"
    cm_path.parent.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(
        y_test,
        y_pred,
        labels=["Class 0", "Class 1"],
        output_path=cm_path,
    )

    csp_patterns_path = None

    if info is not None:
        csp_patterns_path = save_csp_patterns(
            csp=csp,
            info=info,
            output_path="results/motor/csp_patterns.png",
            max_components=4,
        )

    metrics_path = PROJECT_ROOT / "results/motor/motor_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Class 0", "Class 1"],
    )

    return {
        "model_path": str(model_path),
        "confusion_matrix_path": str(cm_path),
        "csp_patterns_path": csp_patterns_path,
        "metrics_path": str(metrics_path),
        "classification_report": report,
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
    }