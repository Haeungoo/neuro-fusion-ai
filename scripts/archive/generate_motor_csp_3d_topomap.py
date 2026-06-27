from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "motor_imagery"
OUTPUT_HTML = RESULT_DIR / "motor_imagery_csp_3d_topomap.html"
OUTPUT_JSON = RESULT_DIR / "motor_imagery_csp_3d_topomap_metadata.json"


# Approximate 10-20 EEG electrode positions on a unit sphere.
# These are not subject-specific digitized positions.
CHANNEL_POSITIONS: Dict[str, Tuple[float, float, float]] = {
    "Fp1": (-0.35, 0.90, 0.25),
    "Fp2": (0.35, 0.90, 0.25),
    "F7": (-0.85, 0.55, 0.10),
    "F3": (-0.45, 0.55, 0.45),
    "Fz": (0.00, 0.60, 0.55),
    "F4": (0.45, 0.55, 0.45),
    "F8": (0.85, 0.55, 0.10),
    "FC3": (-0.55, 0.30, 0.65),
    "FCz": (0.00, 0.35, 0.75),
    "FC4": (0.55, 0.30, 0.65),
    "T7": (-1.00, 0.00, 0.05),
    "C3": (-0.65, 0.00, 0.70),
    "Cz": (0.00, 0.00, 0.90),
    "C4": (0.65, 0.00, 0.70),
    "T8": (1.00, 0.00, 0.05),
    "CP3": (-0.55, -0.35, 0.65),
    "CPz": (0.00, -0.35, 0.75),
    "CP4": (0.55, -0.35, 0.65),
    "P7": (-0.85, -0.60, 0.10),
    "P3": (-0.45, -0.60, 0.45),
    "Pz": (0.00, -0.65, 0.55),
    "P4": (0.45, -0.60, 0.45),
    "P8": (0.85, -0.60, 0.10),
    "O1": (-0.35, -0.90, 0.25),
    "Oz": (0.00, -0.95, 0.30),
    "O2": (0.35, -0.90, 0.25),
}


def normalize_position(position: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Project approximate electrode coordinates onto a unit sphere."""
    x, y, z = position
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0:
        return 0.0, 0.0, 1.0
    return x / norm, y / norm, z / norm


def generate_demo_csp_weights(channel_names: List[str]) -> Dict[str, float]:
    """
    Generate a realistic-looking demo CSP spatial pattern.

    For motor imagery, C3 and C4 regions often show opposite spatial patterns.
    This demo emphasizes left motor cortex around C3 and right motor cortex around C4.
    """
    weights: Dict[str, float] = {}

    for ch in channel_names:
        x, y, z = normalize_position(CHANNEL_POSITIONS[ch])

        # Left/right sensorimotor contrast.
        # Negative x = left scalp side, positive x = right scalp side.
        left_peak = math.exp(-((x + 0.65) ** 2 + (y - 0.00) ** 2) / 0.18)
        right_peak = math.exp(-((x - 0.65) ** 2 + (y - 0.00) ** 2) / 0.18)

        weight = 0.65 * left_peak - 0.65 * right_peak

        # Small central contribution.
        if ch in {"Cz", "FCz", "CPz"}:
            weight *= 0.35

        weights[ch] = float(weight)

    return weights


def make_head_surface() -> go.Surface:
    """Create a translucent half-sphere scalp surface."""
    theta = np.linspace(0, 2 * np.pi, 90)
    phi = np.linspace(0, np.pi / 2, 45)

    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = np.sin(phi_grid) * np.cos(theta_grid)
    y = np.sin(phi_grid) * np.sin(theta_grid)
    z = np.cos(phi_grid)

    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.22,
        colorscale=[[0, "rgb(235, 239, 245)"], [1, "rgb(235, 239, 245)"]],
        showscale=False,
        hoverinfo="skip",
        name="Scalp",
    )


def make_ear_mesh(x_center: float) -> go.Mesh3d:
    """Simple stylized ear mesh for orientation."""
    t = np.linspace(0, 2 * np.pi, 40)
    y = 0.08 * np.cos(t)
    z = 0.28 + 0.22 * np.sin(t)
    x = np.full_like(y, x_center)

    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        opacity=0.25,
        color="lightgray",
        alphahull=0,
        name="Ear",
        hoverinfo="skip",
        showscale=False,
    )


def make_nose_trace() -> go.Scatter3d:
    """Small triangle at the front for head orientation."""
    return go.Scatter3d(
        x=[0.0, -0.08, 0.08, 0.0],
        y=[1.08, 0.95, 0.95, 1.08],
        z=[0.28, 0.20, 0.20, 0.28],
        mode="lines",
        line=dict(color="rgb(90, 90, 90)", width=5),
        name="Nose",
        hoverinfo="skip",
        showlegend=False,
    )


def make_electrode_trace(
    channel_names: List[str],
    weights: Dict[str, float],
) -> go.Scatter3d:
    xs, ys, zs, values, labels = [], [], [], [], []

    for ch in channel_names:
        x, y, z = normalize_position(CHANNEL_POSITIONS[ch])
        xs.append(x)
        ys.append(y)
        zs.append(z)
        values.append(weights[ch])
        labels.append(
            f"Channel: {ch}<br>"
            f"CSP weight: {weights[ch]:+.3f}<br>"
            "Interpretation: model-derived spatial pattern"
        )

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+text",
        text=channel_names,
        textposition="top center",
        hovertext=labels,
        hoverinfo="text",
        marker=dict(
            size=9,
            color=values,
            colorscale="RdBu",
            reversescale=True,
            cmin=-0.7,
            cmax=0.7,
            colorbar=dict(
                title="CSP weight",
                thickness=16,
                len=0.65,
            ),
            line=dict(color="black", width=1),
        ),
        name="EEG electrodes",
    )


def build_figure(channel_names: List[str], weights: Dict[str, float]) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(make_head_surface())
    fig.add_trace(make_ear_mesh(-1.05))
    fig.add_trace(make_ear_mesh(1.05))
    fig.add_trace(make_nose_trace())
    fig.add_trace(make_electrode_trace(channel_names, weights))

    fig.update_layout(
        title=dict(
            text="Motor Imagery CSP 3D Scalp Visualization",
            x=0.02,
            y=0.96,
            font=dict(size=22),
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                eye=dict(x=0.0, y=-2.2, z=1.25),
                center=dict(x=0.0, y=0.0, z=0.25),
            ),
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        annotations=[
            dict(
                text=(
                    "Drag to rotate · Scroll to zoom · Hover over electrodes for channel and CSP weight<br>"
                    "<b>Note:</b> This is a model-derived CSP spatial pattern, not a direct brain activation map."
                ),
                x=0.5,
                y=0.02,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="center",
                font=dict(size=13, color="rgb(80,80,80)"),
                bgcolor="rgba(255, 245, 200, 0.75)",
                bordercolor="rgba(230, 180, 60, 0.8)",
                borderwidth=1,
                borderpad=8,
            )
        ],
    )

    return fig


def write_html(fig: go.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displayModeBar": True,
            "responsive": True,
            "scrollZoom": True,
        },
    )


def write_metadata(
    output_path: Path,
    channel_names: List[str],
    weights: Dict[str, float],
) -> None:
    metadata = {
        "output_html": str(OUTPUT_HTML.relative_to(PROJECT_ROOT)),
        "mode": "demo_csp_spatial_pattern",
        "description": (
            "Interactive 3D scalp visualization of CSP-like spatial pattern weights. "
            "This demo emphasizes opposite C3/C4 sensorimotor patterns for motor imagery."
        ),
        "disclaimer": (
            "This map is a model-derived spatial pattern visualization, not a direct brain activation map."
        ),
        "num_channels": len(channel_names),
        "channels": [
            {
                "name": ch,
                "position": normalize_position(CHANNEL_POSITIONS[ch]),
                "weight": weights[ch],
            }
            for ch in channel_names
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    channel_names = list(CHANNEL_POSITIONS.keys())
    weights = generate_demo_csp_weights(channel_names)

    fig = build_figure(channel_names, weights)

    write_html(fig, OUTPUT_HTML)
    write_metadata(OUTPUT_JSON, channel_names, weights)

    print("Saved 3D CSP topomap:")
    print(f"  HTML: {OUTPUT_HTML}")
    print(f"  JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()