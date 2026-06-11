from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RAW_DIR_CANDIDATES = [
    PROJECT_ROOT / "data" / "seizure_eeg" / "chb01",
    PROJECT_ROOT / "data" / "seizure_eeg",
]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "seizure"
    / "chbmit_multi_file_features.npz"
)


def find_raw_dir() -> Path:
    """
    Find the CHB-MIT raw data directory.

    Expected examples:
        data/seizure_eeg/chb01/chb01_03.edf
        data/seizure_eeg/chb01_03.edf
    """

    for candidate in DEFAULT_RAW_DIR_CANDIDATES:
        if candidate.exists():
            edf_files = list(
                candidate.rglob("*.edf")
            )

            if edf_files:
                return candidate

    raise FileNotFoundError(
        "Could not find CHB-MIT EDF files.\n\n"
        "Expected one of these folders:\n"
        + "\n".join(
            f"  - {path}"
            for path in DEFAULT_RAW_DIR_CANDIDATES
        )
        + "\n\nEach folder should contain .edf files."
    )


def parse_chbmit_summary_file(
    summary_path: Path,
) -> dict[str, list[tuple[float, float]]]:
    """
    Parse CHB-MIT summary text file.

    Returns:
        {
            "chb01_03.edf": [(2996.0, 3036.0)],
            "chb01_04.edf": [],
        }

    The CHB-MIT summary format usually includes lines like:
        File Name: chb01_03.edf
        Number of Seizures in File: 1
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds
    """

    seizure_map: dict[
        str,
        list[tuple[float, float]],
    ] = {}

    current_file: str | None = None
    pending_start: float | None = None

    if not summary_path.exists():
        return seizure_map

    with open(
        summary_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            file_match = re.search(
                r"File Name:\s*(\S+\.edf)",
                line,
                flags=re.IGNORECASE,
            )

            if file_match:
                current_file = file_match.group(1)
                seizure_map.setdefault(
                    current_file,
                    [],
                )
                pending_start = None
                continue

            if current_file is None:
                continue

            start_match = re.search(
                r"Seizure\s+\d*\s*Start Time:\s*"
                r"(\d+(?:\.\d+)?)\s*seconds",
                line,
                flags=re.IGNORECASE,
            )

            if not start_match:
                start_match = re.search(
                    r"Seizure Start Time:\s*"
                    r"(\d+(?:\.\d+)?)\s*seconds",
                    line,
                    flags=re.IGNORECASE,
                )

            if start_match:
                pending_start = float(
                    start_match.group(1)
                )
                continue

            end_match = re.search(
                r"Seizure\s+\d*\s*End Time:\s*"
                r"(\d+(?:\.\d+)?)\s*seconds",
                line,
                flags=re.IGNORECASE,
            )

            if not end_match:
                end_match = re.search(
                    r"Seizure End Time:\s*"
                    r"(\d+(?:\.\d+)?)\s*seconds",
                    line,
                    flags=re.IGNORECASE,
                )

            if (
                end_match
                and pending_start is not None
            ):
                seizure_end = float(
                    end_match.group(1)
                )

                seizure_map.setdefault(
                    current_file,
                    [],
                ).append(
                    (
                        float(pending_start),
                        seizure_end,
                    )
                )

                pending_start = None

    return seizure_map


def build_seizure_interval_map(
    raw_dir: Path,
) -> dict[str, list[tuple[float, float]]]:
    """
    Build seizure interval map for all summary files.
    """

    interval_map: dict[
        str,
        list[tuple[float, float]],
    ] = {}

    summary_files = sorted(
        raw_dir.rglob("*summary*.txt")
    )

    if not summary_files:
        print(
            "Warning: no CHB-MIT summary text files found. "
            "All windows will be labeled as non-seizure."
        )

        return interval_map

    for summary_path in summary_files:
        parsed = parse_chbmit_summary_file(
            summary_path
        )

        interval_map.update(parsed)

    return interval_map


def window_overlaps_seizure(
    window_start_sec: float,
    window_end_sec: float,
    seizure_intervals: list[tuple[float, float]],
) -> int:
    """
    Return 1 if a window overlaps any seizure interval.
    """

    for seizure_start, seizure_end in seizure_intervals:
        overlaps = (
            window_start_sec < seizure_end
            and window_end_sec > seizure_start
        )

        if overlaps:
            return 1

    return 0


def bandpower_features(
    data: np.ndarray,
    sampling_rate: float,
) -> list[float]:
    """
    Calculate EEG bandpower features.

    data shape:
        channels x time
    """

    frequencies, power = welch(
        data,
        fs=sampling_rate,
        nperseg=min(
            int(sampling_rate * 2),
            data.shape[1],
        ),
        axis=1,
    )

    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }

    features: list[float] = []

    total_power = trapezoid(
        power,
        frequencies,
        axis=1,
    ) + 1e-12

    for low, high in bands.values():
        band_mask = np.logical_and(
            frequencies >= low,
            frequencies < high,
        )

        if not np.any(band_mask):
            band_power = np.zeros(
                data.shape[0],
                dtype=np.float32,
            )
        else:
            band_power = trapezoid(
                power[:, band_mask],
                frequencies[band_mask],
                axis=1,
            )

        relative_power = (
            band_power / total_power
        )

        features.extend(
            [
                float(np.mean(band_power)),
                float(np.std(band_power)),
                float(np.mean(relative_power)),
                float(np.std(relative_power)),
            ]
        )

    return features


def time_domain_features(
    data: np.ndarray,
) -> list[float]:
    """
    Calculate simple time-domain EEG features.

    data shape:
        channels x time
    """

    channel_mean = np.mean(
        data,
        axis=1,
    )

    channel_std = np.std(
        data,
        axis=1,
    )

    channel_rms = np.sqrt(
        np.mean(
            data**2,
            axis=1,
        )
    )

    channel_ptp = np.ptp(
        data,
        axis=1,
    )

    line_length = np.sum(
        np.abs(
            np.diff(
                data,
                axis=1,
            )
        ),
        axis=1,
    )

    zero_crossings = np.sum(
        np.diff(
            np.signbit(data),
            axis=1,
        ),
        axis=1,
    )

    features = [
        float(np.mean(channel_mean)),
        float(np.std(channel_mean)),
        float(np.mean(channel_std)),
        float(np.std(channel_std)),
        float(np.mean(channel_rms)),
        float(np.std(channel_rms)),
        float(np.mean(channel_ptp)),
        float(np.std(channel_ptp)),
        float(np.mean(line_length)),
        float(np.std(line_length)),
        float(np.mean(zero_crossings)),
        float(np.std(zero_crossings)),
    ]

    return features


def extract_window_features(
    window_data: np.ndarray,
    sampling_rate: float,
) -> np.ndarray:
    """
    Extract fixed-length features from one EEG window.

    window_data shape:
        channels x time
    """

    window_data = np.asarray(
        window_data,
        dtype=np.float32,
    )

    if window_data.ndim != 2:
        raise ValueError(
            "window_data must have shape "
            f"(channels, time). Got {window_data.shape}."
        )

    # Robust standardization per channel.
    mean = np.mean(
        window_data,
        axis=1,
        keepdims=True,
    )

    std = np.std(
        window_data,
        axis=1,
        keepdims=True,
    ) + 1e-8

    standardized = (
        window_data - mean
    ) / std

    features = []

    features.extend(
        time_domain_features(standardized)
    )

    features.extend(
        bandpower_features(
            data=standardized,
            sampling_rate=sampling_rate,
        )
    )

    return np.asarray(
        features,
        dtype=np.float32,
    )


def load_edf_as_array(
    edf_path: Path,
) -> tuple[np.ndarray, float]:
    """
    Load EDF file using MNE.

    Returns:
        data: channels x time
        sampling_rate
    """

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False,
    )

    raw.pick_types(
        eeg=True,
        exclude="bads",
    )

    if len(raw.ch_names) == 0:
        raise RuntimeError(
            f"No EEG channels found in {edf_path}"
        )

    # Keep frequencies useful for seizure detection.
    raw.filter(
        l_freq=0.5,
        h_freq=45.0,
        verbose=False,
    )

    data = raw.get_data().astype(
        np.float32
    )

    sampling_rate = float(
        raw.info["sfreq"]
    )

    return data, sampling_rate


def process_edf_file(
    edf_path: Path,
    seizure_intervals: list[tuple[float, float]],
    window_seconds: float,
    step_seconds: float,
    max_non_seizure_windows_per_file: int,
) -> tuple[
    list[np.ndarray],
    list[int],
    list[str],
    list[float],
]:
    """
    Process one EDF file into window-level features and labels.
    """

    data, sampling_rate = load_edf_as_array(
        edf_path
    )

    window_samples = int(
        round(window_seconds * sampling_rate)
    )

    step_samples = int(
        round(step_seconds * sampling_rate)
    )

    if window_samples <= 0 or step_samples <= 0:
        raise ValueError(
            "window_samples and step_samples must be positive."
        )

    total_samples = data.shape[1]

    features: list[np.ndarray] = []
    labels: list[int] = []
    file_names: list[str] = []
    window_start_times: list[float] = []

    non_seizure_count = 0

    for start_sample in range(
        0,
        total_samples - window_samples + 1,
        step_samples,
    ):
        end_sample = start_sample + window_samples

        start_sec = (
            start_sample / sampling_rate
        )

        end_sec = (
            end_sample / sampling_rate
        )

        label = window_overlaps_seizure(
            window_start_sec=start_sec,
            window_end_sec=end_sec,
            seizure_intervals=seizure_intervals,
        )

        if (
            label == 0
            and non_seizure_count
            >= max_non_seizure_windows_per_file
        ):
            continue

        window = data[
            :,
            start_sample:end_sample,
        ]

        feature_vector = extract_window_features(
            window_data=window,
            sampling_rate=sampling_rate,
        )

        features.append(feature_vector)
        labels.append(label)
        file_names.append(edf_path.name)
        window_start_times.append(float(start_sec))

        if label == 0:
            non_seizure_count += 1

    return (
        features,
        labels,
        file_names,
        window_start_times,
    )


def main() -> None:
    raw_dir = find_raw_dir()

    output_path = OUTPUT_PATH

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    seizure_interval_map = build_seizure_interval_map(
        raw_dir=raw_dir,
    )

    all_edf_files = sorted(
        raw_dir.rglob("*.edf")
    )

    if not all_edf_files:
        raise FileNotFoundError(
            f"No EDF files found under {raw_dir}"
        )

    seizure_files = [
        path
        for path in all_edf_files
        if seizure_interval_map.get(path.name)
    ]

    non_seizure_files = [
        path
        for path in all_edf_files
        if not seizure_interval_map.get(path.name)
    ]

    edf_files = (
        seizure_files[:4]
        + non_seizure_files[:4]
    )

    print("All EDF files:", len(all_edf_files))
    print("Seizure EDF files:", len(seizure_files))
    print("Non-seizure EDF files:", len(non_seizure_files))
    print("Selected EDF files:")

    for path in edf_files:
        print(
            " ",
            path.name,
            seizure_interval_map.get(path.name, []),
        )

    window_seconds = 5.0
    step_seconds = 2.5
    max_non_seizure_windows_per_file = 300

    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    all_file_names: list[str] = []
    all_window_start_times: list[float] = []

    print("=" * 68)
    print("CHB-MIT seizure feature extraction")
    print("=" * 68)
    print("Raw directory:", raw_dir)
    print("EDF files selected:", len(edf_files))
    print("Window seconds:", window_seconds)
    print("Step seconds:", step_seconds)
    print("=" * 68)

    for edf_path in edf_files:
        intervals = seizure_interval_map.get(
            edf_path.name,
            [],
        )

        print(
            f"Processing {edf_path.name} | "
            f"seizure intervals: {intervals}"
        )

        try:
            (
                features,
                labels,
                file_names,
                window_start_times,
            ) = process_edf_file(
                edf_path=edf_path,
                seizure_intervals=intervals,
                window_seconds=window_seconds,
                step_seconds=step_seconds,
                max_non_seizure_windows_per_file=(
                    max_non_seizure_windows_per_file
                ),
            )

        except Exception as error:
            print(
                f"Skipping {edf_path.name}: {error}"
            )
            continue

        all_features.extend(features)
        all_labels.extend(labels)
        all_file_names.extend(file_names)
        all_window_start_times.extend(
            window_start_times
        )

        print(
            f"  saved windows: {len(labels)} | "
            f"seizure windows: {sum(labels)} | "
            f"non-seizure windows: "
            f"{len(labels) - sum(labels)}"
        )

    if not all_features:
        raise RuntimeError(
            "No features were extracted."
        )

    X = np.vstack(
        all_features
    ).astype(np.float32)

    y = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    file_names_array = np.asarray(
        all_file_names,
        dtype=object,
    )

    window_start_times_array = np.asarray(
        all_window_start_times,
        dtype=np.float32,
    )

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Extracted data contains only one class. "
            "Need both seizure and non-seizure windows.\n"
            "Check that CHB-MIT summary files are present and "
            "that selected EDF files include seizure segments."
        )

    np.savez(
        output_path,
        X=X,
        y=y,
        file_names=file_names_array,
        window_start_times=window_start_times_array,
        window_seconds=float(window_seconds),
        step_seconds=float(step_seconds),
    )

    print()
    print("=" * 68)
    print("Feature extraction complete")
    print("=" * 68)
    print("Output:", output_path)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Seizure windows:", int((y == 1).sum()))
    print("Non-seizure windows:", int((y == 0).sum()))


if __name__ == "__main__":
    main()