from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

def display_path(path) -> str:
    """
    Convert absolute path to project-relative display path.
    """
    from pathlib import Path
    
    path = Path(path)
    
    try:
        relative = path.relative_to(PROJECT_ROOT)
        return str(Path(PROJECT_ROOT.name)/relative)
    except ValueError:
        return str(path)
    

def load_json(relative_path: str) -> dict:
    path = PROJECT_ROOT / relative_path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_demo_assets() -> None:
    """Create simple demo figures if they do not already exist."""
    assets = PROJECT_ROOT / "assets"
    assets.mkdir(exist_ok=True)

    create_demo_mri_overlay(assets / "mri_overlay_demo.png")
    create_demo_eeg_waveform(assets / "eeg_waveform_demo.png")
    create_demo_csp_topomap(assets / "csp_topomap_demo.png")
    create_demo_confusion_matrix(assets / "confusion_matrix_demo.png")
    create_demo_seizure_timeline(assets / "seizure_timeline_demo.png")


def create_demo_mri_overlay(path: Path) -> None:
    if path.exists():
        return

    rng = np.random.default_rng(42)
    x = np.linspace(-1, 1, 160)
    y = np.linspace(-1, 1, 160)
    xx, yy = np.meshgrid(x, y)

    brain = np.exp(-((xx / 0.75) ** 2 + (yy / 0.9) ** 2))
    brain += 0.08 * rng.normal(size=brain.shape)
    brain = np.clip(brain, 0, 1)

    tumor = ((xx - 0.35) ** 2 / 0.08 + (yy + 0.05) ** 2 / 0.14) < 1

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(brain, cmap="gray")
    ax.imshow(np.ma.masked_where(~tumor, tumor), alpha=0.55)
    ax.set_title("MRI Tumor Overlay")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def create_demo_eeg_waveform(path: Path) -> None:
    if path.exists():
        return

    rng = np.random.default_rng(7)
    t = np.linspace(0, 10, 1000)
    channels = []
    for i in range(5):
        sig = 0.4 * np.sin(2 * np.pi * (8 + i) * t)
        sig += 0.2 * np.sin(2 * np.pi * (3 + i) * t)
        sig += 0.3 * rng.normal(size=t.shape)
        channels.append(sig + i * 2.0)

    fig, ax = plt.subplots(figsize=(8, 3))
    for i, sig in enumerate(channels):
        ax.plot(t, sig, linewidth=0.8, label=f"Ch {i+1}")
    ax.axvspan(4.0, 6.5, alpha=0.2, label="Detected event")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude + offset")
    ax.set_title("EEG Waveform Demo")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def create_demo_csp_topomap(path: Path) -> None:
    if path.exists():
        return

    x = np.linspace(-1, 1, 120)
    y = np.linspace(-1, 1, 120)
    xx, yy = np.meshgrid(x, y)
    mask = xx**2 + yy**2 <= 1

    z = np.exp(-((xx + 0.45) ** 2 + yy**2) / 0.18) - np.exp(-((xx - 0.45) ** 2 + yy**2) / 0.18)
    z = np.ma.masked_where(~mask, z)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(z, extent=(-1, 1, -1, 1), origin="lower")
    circle = plt.Circle((0, 0), 1, fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.scatter(np.random.default_rng(1).uniform(-0.8, 0.8, 45),
               np.random.default_rng(2).uniform(-0.8, 0.8, 45),
               s=4)
    ax.set_title("CSP Topomap Demo")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def create_demo_confusion_matrix(path: Path) -> None:
    if path.exists():
        return

    cm = np.array([[0.82, 0.18], [0.21, 0.79]])

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, vmin=0, vmax=1)
    ax.set_xticks([0, 1], labels=["Left", "Right"])
    ax.set_yticks([0, 1], labels=["Left", "Right"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=12)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def create_demo_seizure_timeline(path: Path) -> None:
    if path.exists():
        return

    rng = np.random.default_rng(3)
    t = np.linspace(0, 120, 240)
    prob = 0.08 + 0.06 * rng.normal(size=t.shape)
    prob += 0.8 * np.exp(-((t - 45) ** 2) / 60)
    prob += 0.65 * np.exp(-((t - 90) ** 2) / 80)
    prob = np.clip(prob, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, prob, linewidth=2)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.fill_between(t, 0, prob, where=prob > 0.5, alpha=0.25)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Seizure probability")
    ax.set_title("Seizure Probability Timeline")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    