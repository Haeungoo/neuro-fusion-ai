from __future__ import annotations
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
DATA_DIR = PROJECT_ROOT / 'data'
for path in [MODELS_DIR, RESULTS_DIR, DATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)
