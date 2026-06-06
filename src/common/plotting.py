from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(y_true, y_pred, labels, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax, colorbar=True)
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def save_probability_timeline(probabilities, output_path: str | Path, threshold: float = 0.5) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = np.asarray(probabilities)
    t = np.arange(len(probabilities))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, probabilities, linewidth=2)
    ax.axhline(threshold, linestyle='--', linewidth=1)
    ax.fill_between(t, 0, probabilities, where=probabilities >= threshold, alpha=0.25)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Window index')
    ax.set_ylabel('Seizure probability')
    ax.set_title('Seizure Probability Timeline')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def save_mri_overlay(image_2d, mask_2d, output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_2d = np.asarray(image_2d)
    mask_2d = np.asarray(mask_2d).astype(bool)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image_2d, cmap='gray')
    ax.imshow(np.ma.masked_where(~mask_2d, mask_2d), alpha=0.55)
    ax.axis('off')
    ax.set_title('MRI Tumor Overlay')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)

def save_eeg_waveform(
    eeg,
    fs: float,
    output_path,
    max_channels: int = 6,
) -> str:
    """
    Save EEG waveform plot.

    Parameters
    ----------
    eeg:
        EEG signal with shape (n_channels, n_times).

    fs:
        Sampling frequency.

    output_path:
        Path where the waveform image will be saved.

    max_channels:
        Maximum number of EEG channels to plot.

    Returns
    -------
    str:
        Saved image path.
    """
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eeg = np.asarray(eeg, dtype=float)

    if eeg.ndim != 2:
        raise ValueError(
            f"eeg must be 2D with shape (n_channels, n_times). Got shape {eeg.shape}"
        )

    n_channels = min(max_channels, eeg.shape[0])
    t = np.arange(eeg.shape[1]) / fs

    fig, ax = plt.subplots(figsize=(9, 3.5))

    offset = 0.0

    for ch in range(n_channels):
        signal = eeg[ch]

        # Normalize each channel for visualization
        signal = signal / (np.std(signal) + 1e-8)

        ax.plot(
            t,
            signal + offset,
            linewidth=0.8,
            label=f"Ch {ch + 1}",
        )

        offset += 4.0

    ax.set_title("EEG Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels + offset")
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return str(output_path)

def save_training_curve(history: dict, output_path) -> str:
    """
    Save training/validation loss curve.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train loss")

    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MRI U-Net Training Curve")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return str(output_path)