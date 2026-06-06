from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.seizure_detection.windowing import make_sliding_windows, label_windows_from_intervals
from src.seizure_detection.features import extract_features_from_windows
from src.seizure_detection.train import train_random_forest_seizure


def create_synthetic_eeg(fs=128, duration_sec=300, n_channels=4):
    rng = np.random.default_rng(42)
    t = np.arange(int(fs * duration_sec)) / fs
    eeg = 0.3 * rng.normal(size=(n_channels, len(t)))
    for ch in range(n_channels):
        eeg[ch] += 0.1 * np.sin(2 * np.pi * (8 + ch) * t)
    seizure_intervals = [(80, 105), (210, 235)]
    for start, end in seizure_intervals:
        idx = (t >= start) & (t <= end)
        for ch in range(n_channels):
            eeg[ch, idx] += 2.0 * np.sin(2 * np.pi * 5 * t[idx])
            eeg[ch, idx] += 0.8 * rng.normal(size=idx.sum())
    return eeg, seizure_intervals

def main():
    fs = 128
    eeg, seizure_intervals = create_synthetic_eeg(fs=fs)
    windows, starts = make_sliding_windows(eeg, fs, window_sec=5.0, step_sec=2.5)
    y = label_windows_from_intervals(starts, window_sec=5.0, seizure_intervals=seizure_intervals)
    X_features = extract_features_from_windows(windows, fs)
    print('windows shape:', windows.shape)
    print('features shape:', X_features.shape)
    print('label counts:', np.bincount(y))
    result = train_random_forest_seizure(X_features, y)
    print('Training complete.')
    for k, v in result.items(): print(f'{k}: {v}')
    
if __name__ == '__main__': main()
