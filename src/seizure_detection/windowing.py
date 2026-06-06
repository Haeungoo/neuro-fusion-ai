from __future__ import annotations
import numpy as np


def make_sliding_windows(eeg: np.ndarray, fs: float, window_sec: float = 5.0, step_sec: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    eeg = np.asarray(eeg)
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    windows, starts = [], []
    for start in range(0, eeg.shape[1] - win + 1, step):
        windows.append(eeg[:, start:start+win])
        starts.append(start / fs)
    return np.asarray(windows), np.asarray(starts)


def label_windows_from_intervals(starts_sec: np.ndarray, window_sec: float, seizure_intervals: list[tuple[float, float]], overlap_threshold: float = 0.25) -> np.ndarray:
    labels = []
    for start in starts_sec:
        end = start + window_sec
        is_seizure = False
        for sz_start, sz_end in seizure_intervals:
            overlap = max(0.0, min(end, sz_end) - max(start, sz_start))
            if overlap / window_sec >= overlap_threshold:
                is_seizure = True
                break
        labels.append(int(is_seizure))
    return np.asarray(labels)
