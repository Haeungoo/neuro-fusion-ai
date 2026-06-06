from __future__ import annotations
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy


def bandpower(signal: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 2)))
    idx = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(idx): return 0.0
    return float(np.trapezoid(psd[idx], freqs[idx]))


def line_length(signal: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(signal))))


def hjorth_parameters(signal: np.ndarray) -> tuple[float, float, float]:
    signal = np.asarray(signal)
    d1 = np.diff(signal)
    d2 = np.diff(d1)
    var0 = np.var(signal) + 1e-12
    var1 = np.var(d1) + 1e-12
    var2 = np.var(d2) + 1e-12
    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility
    return float(activity), float(mobility), float(complexity)


def signal_entropy(signal: np.ndarray, bins: int = 30) -> float:
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist + 1e-12
    return float(scipy_entropy(hist))


def extract_window_features(window: np.ndarray, fs: float) -> np.ndarray:
    window = np.asarray(window)
    features = []
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    for ch in range(window.shape[0]):
        sig = window[ch]
        features.extend([np.mean(sig), np.std(sig), np.max(sig)-np.min(sig), line_length(sig), signal_entropy(sig)])
        features.extend(hjorth_parameters(sig))
        for band in bands:
            features.append(bandpower(sig, fs, band))
    return np.array(features, dtype=float)


def extract_features_from_windows(windows: np.ndarray, fs: float) -> np.ndarray:
    return np.vstack([extract_window_features(w, fs) for w in windows])
