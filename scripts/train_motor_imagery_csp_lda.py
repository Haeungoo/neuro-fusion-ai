from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.motor_imagery.metrics import binary_classification_metrics


try:
    from src.common.paths import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "motor_imagery"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_PATH = (
    MODELS_DIR
    / "motor_imagery_csp_lda.joblib"
)


def generate_synthetic_motor_imagery_data(
    num_trials_per_class: int = 120,
    num_channels: int = 8,
    num_samples: int = 256,
    sampling_rate: int = 128,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG-like motor imagery data.

    Label convention:
    0 = left-hand motor imagery
    1 = right-hand motor imagery

    Shape:
    X = (num_trials, num_channels, num_samples)
    y = (num_trials,)
    """

    rng = np.random.default_rng(random_seed)

    total_trials = num_trials_per_class * 2

    X = np.zeros(
        (
            total_trials,
            num_channels,
            num_samples,
        ),
        dtype=np.float64,
    )

    y = np.zeros(
        total_trials,
        dtype=np.int64,
    )

    time = np.arange(num_samples) / sampling_rate

    mu_rhythm = np.sin(
        2 * np.pi * 10 * time,
    )

    beta_rhythm = np.sin(
        2 * np.pi * 20 * time,
    )

    for trial_index in range(total_trials):
        if trial_index < num_trials_per_class:
            label = 0
        else:
            label = 1

        y[trial_index] = label

        trial = rng.normal(
            0,
            0.25,
            size=(num_channels, num_samples),
        )

        for channel_index in range(num_channels):
            trial[channel_index] += (
                0.35 * mu_rhythm
                + 0.15 * beta_rhythm
            )

        if label == 0:
            # Left-hand imagery:
            # stronger pattern on right sensorimotor-like channels
            trial[4:8] += (
                0.65 * mu_rhythm
                + 0.30 * beta_rhythm
            )

            trial[0:4] += (
                0.15 * mu_rhythm
            )

        else:
            # Right-hand imagery:
            # stronger pattern on left sensorimotor-like channels
            trial[0:4] += (
                0.65 * mu_rhythm
                + 0.30 * beta_rhythm
            )

            trial[4:8] += (
                0.15 * mu_rhythm
            )

        X[trial_index] = trial

    shuffled_indices = rng.permutation(total_trials)

    return (
        X[shuffled_indices],
        y[shuffled_indices],
    )


def covariance_matrix(
    trial: np.ndarray,
) -> np.ndarray:
    """
    Calculate normalized covariance matrix for one EEG trial.

    trial shape:
    (num_channels, num_samples)
    """

    covariance = trial @ trial.T

    trace = np.trace(covariance)

    if trace == 0:
        return covariance

    return covariance / trace


def fit_csp(
    X: np.ndarray,
    y: np.ndarray,
    num_components: int = 4,
) -> np.ndarray:
    """
    Fit a simple binary CSP spatial filter.

    CSP finds spatial filters that maximize variance differences
    between two classes.

    Returns:
    filters shape = (num_components, num_channels)
    """

    class_0_trials = X[y == 0]
    class_1_trials = X[y == 1]

    cov_0 = np.mean(
        [
            covariance_matrix(trial)
            for trial in class_0_trials
        ],
        axis=0,
    )

    cov_1 = np.mean(
        [
            covariance_matrix(trial)
            for trial in class_1_trials
        ],
        axis=0,
    )

    composite_cov = cov_0 + cov_1

    eigenvalues, eigenvectors = np.linalg.eigh(composite_cov)

    sort_order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[sort_order]
    eigenvectors = eigenvectors[:, sort_order]

    whitening_matrix = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
        @ eigenvectors.T
    )

    whitened_cov_0 = (
        whitening_matrix
        @ cov_0
        @ whitening_matrix.T
    )

    csp_eigenvalues, csp_eigenvectors = np.linalg.eigh(
        whitened_cov_0,
    )

    sort_order = np.argsort(csp_eigenvalues)[::-1]

    csp_eigenvectors = csp_eigenvectors[:, sort_order]

    spatial_filters = (
        csp_eigenvectors.T
        @ whitening_matrix
    )

    half_components = num_components // 2

    selected_filters = np.vstack(
        [
            spatial_filters[:half_components],
            spatial_filters[-half_components:],
        ]
    )

    return selected_filters


def transform_csp(
    X: np.ndarray,
    spatial_filters: np.ndarray,
) -> np.ndarray:
    """
    Transform EEG trials into CSP log-variance features.

    Input:
    X shape = (num_trials, num_channels, num_samples)

    Output:
    features shape = (num_trials, num_components)
    """

    features = []

    for trial in X:
        projected = spatial_filters @ trial

        variances = np.var(
            projected,
            axis=1,
        )

        normalized_variances = variances / (
            np.sum(variances) + 1e-10
        )

        log_variances = np.log(
            normalized_variances + 1e-10,
        )

        features.append(log_variances)

    return np.asarray(features)


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def save_predictions_csv(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trial_index",
                "y_true",
                "y_pred",
                "probability_right_hand",
            ],
        )

        writer.writeheader()

        for index, (
            true_label,
            predicted_label,
            probability,
        ) in enumerate(
            zip(
                y_true,
                y_pred,
                y_probability,
            )
        ):
            writer.writerow(
                {
                    "trial_index": index,
                    "y_true": int(true_label),
                    "y_pred": int(predicted_label),
                    "probability_right_hand": float(probability),
                }
            )


def save_confusion_matrix_image(
    path: Path,
    matrix: list[list[int]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_array = np.asarray(matrix)

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix_array)

    plt.title("Motor Imagery Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    labels = [
        "Left hand",
        "Right hand",
    ]

    plt.xticks(
        [0, 1],
        labels,
        rotation=20,
        ha="right",
    )

    plt.yticks(
        [0, 1],
        labels,
    )

    for row in range(matrix_array.shape[0]):
        for col in range(matrix_array.shape[1]):
            plt.text(
                col,
                row,
                str(matrix_array[row, col]),
                ha="center",
                va="center",
                fontsize=14,
            )

    plt.colorbar(
        fraction=0.046,
        pad=0.04,
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=160,
    )
    plt.close()


def train_motor_imagery_csp_lda() -> dict[str, Any]:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y = generate_synthetic_motor_imagery_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    num_csp_components = 4

    spatial_filters = fit_csp(
        X_train,
        y_train,
        num_components=num_csp_components,
    )

    X_train_features = transform_csp(
        X_train,
        spatial_filters,
    )

    X_test_features = transform_csp(
        X_test,
        spatial_filters,
    )

    classifier = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "lda",
                LinearDiscriminantAnalysis(),
            ),
        ]
    )

    classifier.fit(
        X_train_features,
        y_train,
    )

    y_pred = classifier.predict(
        X_test_features,
    )

    y_probability = classifier.predict_proba(
        X_test_features,
    )[:, 1]

    metrics = binary_classification_metrics(
        y_test,
        y_pred,
    )

    metrics.update(
        {
            "model": "CSP + LDA",
            "dataset": "synthetic_motor_imagery_eeg",
            "num_csp_components": int(num_csp_components),
            "num_channels": int(X.shape[1]),
            "num_samples_per_trial": int(X.shape[2]),
            "num_train_trials": int(len(y_train)),
            "num_test_trials": int(len(y_test)),
        }
    )

    save_json(
        RESULTS_DIR / "motor_imagery_metrics.json",
        metrics,
    )

    save_predictions_csv(
        RESULTS_DIR / "motor_imagery_predictions.csv",
        y_test,
        y_pred,
        y_probability,
    )

    save_confusion_matrix_image(
        RESULTS_DIR / "motor_imagery_confusion_matrix.png",
        metrics["confusion_matrix"],
    )

    model_bundle = {
        "model": classifier,
        "spatial_filters": spatial_filters,
        "num_csp_components": num_csp_components,
        "label_0": "left_hand_imagery",
        "label_1": "right_hand_imagery",
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    return metrics


def main() -> None:
    metrics = train_motor_imagery_csp_lda()

    print("[done] Motor imagery CSP + LDA training completed.")
    print(f"[metrics] {RESULTS_DIR / 'motor_imagery_metrics.json'}")
    print(f"[predictions] {RESULTS_DIR / 'motor_imagery_predictions.csv'}")
    print(f"[confusion matrix] {RESULTS_DIR / 'motor_imagery_confusion_matrix.png'}")
    print(f"[model] {MODEL_PATH}")
    print(
        "[accuracy]",
        round(
            metrics["accuracy"],
            4,
        ),
    )


if __name__ == "__main__":
    main()