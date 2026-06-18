from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np


try:
    from src.common.paths import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]


RESULTS_DIR = PROJECT_ROOT / "results" / "seizure"

POSSIBLE_DATA_DIRS = [
    PROJECT_ROOT / "data" / "seizure_eeg",
    PROJECT_ROOT / "data" / "seizure",
    PROJECT_ROOT / "data" / "chbmit",
    PROJECT_ROOT / "data" / "chb_mit",
]


def find_first_edf_file() -> Path | None:
    for data_dir in POSSIBLE_DATA_DIRS:
        if not data_dir.exists():
            continue

        edf_files = sorted(data_dir.rglob("*.edf"))

        if edf_files:
            return edf_files[0]

    return None


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(float)

    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:
        return signal - mean

    return (signal - mean) / std


def min_max_scale(values: np.ndarray) -> np.ndarray:
    min_value = float(np.min(values))
    max_value = float(np.max(values))

    if max_value == min_value:
        return np.zeros_like(values)

    return (values - min_value) / (max_value - min_value)


def create_probability_like_score(
    data: np.ndarray,
    sfreq: float,
    window_seconds: float = 2.0,
    step_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a seizure-likeness timeline from EEG signal energy and line length.

    This is a visualization score. If your existing seizure model already
    produces true probabilities, this function can later be replaced with
    model predict_proba outputs.
    """

    window_size = int(window_seconds * sfreq)
    step_size = int(step_seconds * sfreq)

    n_samples = data.shape[1]

    times = []
    scores = []

    for start in range(0, n_samples - window_size, step_size):
        stop = start + window_size
        window = data[:, start:stop]

        rms_energy = np.mean(np.sqrt(np.mean(window**2, axis=1)))
        line_length = np.mean(np.sum(np.abs(np.diff(window, axis=1)), axis=1))

        score = 0.65 * rms_energy + 0.35 * line_length

        center_time = (start + stop) / 2 / sfreq

        times.append(center_time)
        scores.append(score)

    scores_array = np.asarray(scores)
    times_array = np.asarray(times)

    scaled_scores = min_max_scale(scores_array)

    return times_array, scaled_scores


def load_real_eeg_segment(
    edf_path: Path,
    duration_seconds: float = 60.0,
    max_channels: int = 4,
) -> tuple[np.ndarray, np.ndarray, float, list[str], dict[str, Any]]:
    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False,
    )

    try:
        raw.pick_types(
            eeg=True,
            meg=False,
            stim=False,
            eog=False,
            exclude="bads",
        )
    except Exception:
        pass

    if len(raw.ch_names) == 0:
        raise RuntimeError("No EEG channels were found in the EDF file.")

    selected_channels = raw.ch_names[:max_channels]

    sfreq = float(raw.info["sfreq"])
    max_samples = int(duration_seconds * sfreq)

    data = raw.get_data(
        picks=selected_channels,
        start=0,
        stop=min(max_samples, raw.n_times),
    )

    times = np.arange(data.shape[1]) / sfreq

    normalized_data = np.vstack(
        [
            normalize_signal(channel_data)
            for channel_data in data
        ]
    )

    metadata = {
        "source": "real_edf",
        "edf_path": str(edf_path),
        "duration_seconds": float(times[-1]) if len(times) > 0 else 0.0,
        "sampling_frequency": sfreq,
        "channels": selected_channels,
        "num_channels": len(selected_channels),
        "num_samples": int(data.shape[1]),
        "note": (
            "Waveform is loaded from a local EDF file. The probability timeline "
            "is an energy-based seizure-likeness visualization score unless "
            "replaced by model predict_proba outputs later."
        ),
    }

    return normalized_data, times, sfreq, selected_channels, metadata


def create_synthetic_fallback_segment(
    duration_seconds: float = 60.0,
    sfreq: float = 256.0,
    max_channels: int = 4,
) -> tuple[np.ndarray, np.ndarray, float, list[str], dict[str, Any]]:
    """
    Fallback only: creates a synthetic EEG-like signal when no local EDF exists.
    """

    rng = np.random.default_rng(42)

    times = np.arange(0, duration_seconds, 1 / sfreq)

    data = []

    for channel_index in range(max_channels):
        alpha = np.sin(2 * np.pi * 10 * times)
        theta = 0.6 * np.sin(2 * np.pi * 5 * times + channel_index)
        noise = 0.35 * rng.normal(size=len(times))

        seizure_burst = np.zeros_like(times)
        burst_mask = (times >= 24) & (times <= 37)
        seizure_burst[burst_mask] = (
            2.5
            * np.sin(2 * np.pi * 18 * times[burst_mask])
            * np.hanning(np.sum(burst_mask))
        )

        signal = alpha + theta + noise + seizure_burst
        data.append(normalize_signal(signal))

    data_array = np.vstack(data)

    channel_names = [
        f"EEG {index + 1}"
        for index in range(max_channels)
    ]

    metadata = {
        "source": "synthetic_fallback",
        "duration_seconds": duration_seconds,
        "sampling_frequency": sfreq,
        "channels": channel_names,
        "num_channels": len(channel_names),
        "num_samples": int(data_array.shape[1]),
        "note": (
            "No local EDF file was found. A synthetic EEG-like fallback signal "
            "was generated for dashboard visualization only."
        ),
    }

    return data_array, times, sfreq, channel_names, metadata


def save_waveform_plot(
    path: Path,
    data: np.ndarray,
    times: np.ndarray,
    channel_names: list[str],
    probability_times: np.ndarray,
    probability_scores: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(13, 5))

    vertical_spacing = 5.0

    for index, channel_data in enumerate(data):
        offset = index * vertical_spacing
        plt.plot(
            times,
            channel_data + offset,
            linewidth=0.8,
            label=channel_names[index],
        )

    high_score_mask = probability_scores >= 0.65

    if np.any(high_score_mask):
        high_times = probability_times[high_score_mask]

        plt.axvspan(
            float(np.min(high_times)),
            float(np.max(high_times)),
            alpha=0.18,
            label="High seizure-likeness segment",
        )

    plt.title("EEG Waveform Preview")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized EEG channels")

    y_ticks = [
        index * vertical_spacing
        for index in range(len(channel_names))
    ]

    plt.yticks(
        y_ticks,
        channel_names,
    )

    plt.legend(
        loc="upper right",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=170,
    )

    plt.close()


def save_probability_timeline_plot(
    path: Path,
    probability_times: np.ndarray,
    probability_scores: np.ndarray,
    threshold: float = 0.65,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(13, 4))

    plt.plot(
        probability_times,
        probability_scores,
        linewidth=2.0,
        label="Seizure-likeness score",
    )

    plt.axhline(
        threshold,
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold {threshold:.2f}",
    )

    plt.fill_between(
        probability_times,
        probability_scores,
        threshold,
        where=probability_scores >= threshold,
        alpha=0.25,
        interpolate=True,
        label="Predicted high-risk region",
    )

    plt.ylim(0.0, 1.05)

    plt.title("Seizure Probability Timeline")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Probability-like score")

    plt.legend(
        loc="upper right",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=170,
    )

    plt.close()


def save_metadata(
    path: Path,
    metadata: dict[str, Any],
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
            metadata,
            file,
            indent=2,
        )


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    edf_path = find_first_edf_file()

    if edf_path is not None:
        print("[seizure visualization] EDF found:", edf_path)

        data, times, sfreq, channel_names, metadata = load_real_eeg_segment(
            edf_path=edf_path,
            duration_seconds=60.0,
            max_channels=4,
        )

    else:
        print(
            "[seizure visualization] No EDF file found. "
            "Using synthetic fallback signal."
        )

        data, times, sfreq, channel_names, metadata = create_synthetic_fallback_segment(
            duration_seconds=60.0,
            sfreq=256.0,
            max_channels=4,
        )

    probability_times, probability_scores = create_probability_like_score(
        data=data,
        sfreq=sfreq,
        window_seconds=2.0,
        step_seconds=1.0,
    )

    waveform_path = RESULTS_DIR / "seizure_eeg_waveform.png"
    probability_timeline_path = RESULTS_DIR / "seizure_probability_timeline.png"
    metadata_path = RESULTS_DIR / "seizure_visualization_metadata.json"

    save_waveform_plot(
        path=waveform_path,
        data=data,
        times=times,
        channel_names=channel_names,
        probability_times=probability_times,
        probability_scores=probability_scores,
    )

    save_probability_timeline_plot(
        path=probability_timeline_path,
        probability_times=probability_times,
        probability_scores=probability_scores,
        threshold=0.65,
    )

    metadata.update(
        {
            "waveform_image": str(waveform_path),
            "probability_timeline_image": str(probability_timeline_path),
            "probability_threshold": 0.65,
            "score_type": "energy_based_probability_like_score",
        }
    )

    save_metadata(
        path=metadata_path,
        metadata=metadata,
    )

    print("[saved]", waveform_path)
    print("[saved]", probability_timeline_path)
    print("[saved]", metadata_path)


if __name__ == "__main__":
    main()