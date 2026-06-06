from __future__ import annotations

import json
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_metrics_from_model(model_path: Path, output_path: Path) -> None:
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    bundle = joblib.load(model_path)
    metrics = bundle.get("metrics", {})

    if not metrics:
        print(f"No metrics found inside: {model_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved: {output_path}")
    print(metrics)


def main() -> None:
    save_metrics_from_model(
        PROJECT_ROOT / "models/seizure_rf.pkl",
        PROJECT_ROOT / "results/seizure/seizure_metrics.json",
    )

    save_metrics_from_model(
        PROJECT_ROOT / "models/motor_csp_lda.pkl",
        PROJECT_ROOT / "results/motor/motor_metrics.json",
    )


if __name__ == "__main__":
    main()