from __future__ import annotations

from pathlib import Path
import re

import mne
import numpy as np


def parse_chbmit_summary(summary_path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """
    Parse CHB-MIT summary file and return seizure intervals by EDF file.

    Returns
    -------
    dict:
        {
            "chb01_03.edf": [(2996.0, 3036.0)],
            "chb01_04.edf": [(1467.0, 1494.0)],
            ...
        }
    """

    summary_path = Path(summary_path)

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    text = summary_path.read_text(errors="ignore")

    seizure_map: dict[str, list[tuple[float, float]]] = {}

    current_file: str | None = None
    pending_start: float | None = None

    for line in text.splitlines():
        line = line.strip()

        file_match = re.match(r"File Name:\s*(.+\.edf)", line)
        if file_match:
            current_file = file_match.group(1).strip()
            seizure_map.setdefault(current_file, [])
            pending_start = None
            continue

        if current_file is None:
            continue

        start_match = re.search(r"Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds", line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue

        end_match = re.search(r"Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds", line)
        if end_match and pending_start is not None:
            end_time = float(end_match.group(1))
            seizure_map[current_file].append((pending_start, end_time))
            pending_start = None

    # keep only files with seizures
    seizure_map = {
        file_name: intervals
        for file_name, intervals in seizure_map.items()
        if len(intervals) > 0
    }

    return seizure_map


def load_chbmit_edf(
    edf_path: str | Path,
    max_channels: int = 8,
    l_freq: float = 0.5,
    h_freq: float = 45.0,
) -> tuple[np.ndarray, float, list[str]]:
    """
    Load one CHB-MIT EDF file and return EEG array.

    Returns
    -------
    eeg:
        shape (n_channels, n_times)

    fs:
        sampling frequency

    ch_names:
        channel names
    """

    edf_path = Path(edf_path)

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    raw.pick_types(eeg=True)

    if len(raw.ch_names) > max_channels:
        raw.pick_channels(raw.ch_names[:max_channels])

    raw.filter(l_freq, h_freq, fir_design="firwin", verbose=False)

    eeg = raw.get_data()
    fs = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)

    return eeg, fs, ch_names


def list_available_seizure_edfs(
    data_dir: str | Path,
    seizure_map: dict[str, list[tuple[float, float]]],
) -> list[Path]:
    """
    Return EDF files that exist locally and have seizure annotations.
    """

    data_dir = Path(data_dir)

    edf_paths = []

    for file_name in sorted(seizure_map.keys()):
        edf_path = data_dir / file_name

        if edf_path.exists():
            edf_paths.append(edf_path)
        else:
            print(f"Warning: missing EDF file: {edf_path}")

    return edf_paths
