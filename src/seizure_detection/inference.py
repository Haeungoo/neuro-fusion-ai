from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np

from src.common.paths import PROJECT_ROOT
from src.common.plotting import save_probability_timeline, save_eeg_waveform
from src.seizure_detection.features import extract_features_from_windows
from src.seizure_detection.windowing import make_sliding_windows


def run_seizure_inference(
    eeg: np.ndarray,
    fs: float,
    model_path: str | Path = "models/seizure_rf.pkl",
    window_sec: float = 5.0,
    step_sec: float = 2.5,
    threshold: float = 0.5,
) -> dict:
    """
    Real seizure inference function.

    Input:
        eeg: EEG signal, shape (n_channels, n_times)
        fs: sampling frequency

    Output:
        seizure probability timeline and prediction result
    """
    model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Seizure model not found: {model_path}\n"
            "Please run:\n"
            "python -m scripts.train_seizure_demo"
        )

    eeg = np.asarray(eeg, dtype=float)

    if eeg.ndim != 2:
        raise ValueError(
            f"eeg must be 2D with shape (n_channels, n_times). Got shape {eeg.shape}"
        )

    bundle = joblib.load(model_path)
    model = bundle["model"]

    windows, starts_sec = make_sliding_windows(
        eeg=eeg,
        fs=fs,
        window_sec=window_sec,
        step_sec=step_sec,
    )

    X_features = extract_features_from_windows(windows, fs)

    probabilities = model.predict_proba(X_features)[:, 1]
    detected = probabilities >= threshold

    timeline_path = PROJECT_ROOT / "results/seizure/seizure_probability_timeline.png"
    waveform_path = PROJECT_ROOT / "results/seizure/eeg_waveform_input.png"

    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    waveform_path.parent.mkdir(parents=True, exist_ok=True)

    save_probability_timeline(
        probabilities=probabilities,
        output_path=timeline_path,
        threshold=threshold,
    )

    save_eeg_waveform(
        eeg=eeg,
        fs=fs,
        output_path=waveform_path,
    )

    if np.any(detected):
        first_idx = int(np.argmax(detected))
        first_detection_sec = float(starts_sec[first_idx])
    else:
        first_detection_sec = None

    return {
        "prediction": "Seizure detected" if bool(np.any(detected)) else "No seizure detected",
        "num_windows": int(len(windows)),
        "num_detected_windows": int(np.sum(detected)),
        "max_probability": float(np.max(probabilities)),
        "first_detection_sec": first_detection_sec,
        "timeline_path": str(timeline_path),
        "waveform_path": str(waveform_path),
        "probabilities": probabilities.tolist(),
        "window_starts_sec": starts_sec.tolist(),
        "model_metrics": bundle.get("metrics", {}),
    }


def create_synthetic_seizure_eeg(
    fs: float = 128,
    duration_sec: float = 120,
    n_channels: int = 4,
    seizure_interval: tuple[float, float] = (45, 65),
    seed: int = 42,
) -> np.ndarray:
    """
    Create synthetic EEG-like signal with one seizure-like segment.
    This is not real clinical EEG. It is only for testing.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration_sec)) / fs

    eeg = 0.3 * rng.normal(size=(n_channels, len(t)))

    for ch in range(n_channels):
        eeg[ch] += 0.1 * np.sin(2 * np.pi * (8 + ch) * t)
        eeg[ch] += 0.05 * np.sin(2 * np.pi * 20 * t)

    idx = (t >= seizure_interval[0]) & (t <= seizure_interval[1])

    for ch in range(n_channels):
        eeg[ch, idx] += 2.0 * np.sin(2 * np.pi * 5 * t[idx])
        eeg[ch, idx] += 0.8 * rng.normal(size=idx.sum())

    return eeg


def run_seizure_dashboard_demo() -> dict:
    """
    Dashboard-safe seizure demo.

    This can be called without arguments.
    It requires models/seizure_rf.pkl.
    """
    fs = 128

    eeg = create_synthetic_seizure_eeg(
        fs=fs,
        duration_sec=120,
        n_channels=4,
        seizure_interval=(45, 65),
        seed=42,
    )

    return run_seizure_inference(
        eeg=eeg,
        fs=fs,
        model_path="models/seizure_rf.pkl",
        window_sec=5.0,
        step_sec=2.5,
        threshold=0.5,
    )