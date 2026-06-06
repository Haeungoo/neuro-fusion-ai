from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np

from src.common.paths import PROJECT_ROOT


def run_motor_inference(
    epoch: np.ndarray,
    model_path: str | Path = "models/motor_csp_lda.pkl",
) -> dict:
    """
    Real motor imagery inference function.

    Parameters
    ----------
    epoch:
        EEG epoch.
        Shape can be:
        - (n_channels, n_times)
        - (1, n_channels, n_times)

    model_path:
        Path to trained CSP + LDA model.

    Returns
    -------
    dict:
        Prediction result for dashboard or script use.
    """

    model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Motor imagery model not found: {model_path}\n"
            "Please run:\n"
            "python -m scripts.train_motor_imagery"
        )

    bundle = joblib.load(model_path)

    if "csp" not in bundle or "lda" not in bundle:
        raise KeyError(
            "The motor model bundle must contain 'csp' and 'lda'. "
            "Please retrain with `python -m scripts.train_motor_imagery`."
        )

    csp = bundle["csp"]
    lda = bundle["lda"]

    epoch = np.asarray(epoch, dtype=float)

    if epoch.ndim == 2:
        epoch = epoch[None, :, :]

    if epoch.ndim != 3:
        raise ValueError(
            f"epoch must have shape (n_channels, n_times) or "
            f"(1, n_channels, n_times). Got shape {epoch.shape}"
        )

    features = csp.transform(epoch)
    pred = int(lda.predict(features)[0])

    confidence = None
    if hasattr(lda, "predict_proba"):
        probs = lda.predict_proba(features)[0]
        confidence = float(np.max(probs))

    label_map = {
        0: "Class 0 / T1 imagery",
        1: "Class 1 / T2 imagery",
    }

    topomap_path = PROJECT_ROOT / "results/motor/csp_patterns.png"
    confusion_matrix_path = PROJECT_ROOT / "results/motor/confusion_matrix_motor.png"

    return {
        "prediction": label_map.get(pred, str(pred)),
        "class_id": pred,
        "confidence": confidence,
        "model_metrics": bundle.get("metrics", {}),
        "topomap_path": str(topomap_path),
        "confusion_matrix_path": str(confusion_matrix_path),
    }


def run_motor_dashboard_demo() -> dict:
    """
    Dashboard-safe motor imagery demo.

    This function can be called without arguments from Streamlit.

    It requires:
        models/motor_csp_lda.pkl

    If the model does not exist, train it first:

        python -m scripts.train_motor_imagery
    """

    model_path = PROJECT_ROOT / "models/motor_csp_lda.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "models/motor_csp_lda.pkl not found.\n"
            "Please run:\n"
            "python -m scripts.train_motor_imagery"
        )

    bundle = joblib.load(model_path)

    n_channels = int(bundle.get("n_channels", 64))
    n_times = int(bundle.get("n_times", 481))

    rng = np.random.default_rng(7)
    epoch = rng.normal(size=(n_channels, n_times)).astype(float)

    result = run_motor_inference(
        epoch=epoch,
        model_path="models/motor_csp_lda.pkl",
    )

    # Ensure dashboard compatibility
    result.setdefault(
        "topomap_path",
        str(PROJECT_ROOT / "results/motor/csp_patterns.png"),
    )
    result.setdefault(
        "confusion_matrix_path",
        str(PROJECT_ROOT / "results/motor/confusion_matrix_motor.png"),
    )

    return result


def run_motor_demo_inference() -> dict:
    """
    Backward-compatible alias for older dashboard code.

    Some earlier versions of motor_page.py may import:

        run_motor_demo_inference

    This function keeps that old import working.
    """

    return run_motor_dashboard_demo()