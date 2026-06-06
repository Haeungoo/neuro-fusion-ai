from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import files, motor, mri, seizure
from src.common.paths import PROJECT_ROOT


app = FastAPI(
    title="NeuroFusion-AI API",
    description="FastAPI backend for NeuroFusion-AI.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/media/results",
    StaticFiles(directory=PROJECT_ROOT / "results"),
    name="results",
)


@app.get("/")
def root() -> dict:
    return {
        "project": "NeuroFusion-AI",
        "version": "0.3.0",
        "message": "NeuroFusion-AI backend is running.",
    }


@app.get("/api/overview")
def overview() -> dict:
    return {
        "project": "NeuroFusion-AI",
        "description": (
            "Multimodal neuroscience AI platform integrating MRI tumor "
            "segmentation, EEG seizure detection, and motor imagery BCI."
        ),
        "version": "0.3.0",
        "backend": "FastAPI",
        "current_frontend": "Next.js",
        "modules": {
            "mri": {
                "task": "Brain tumor segmentation",
                "model": "2D U-Net",
                "dataset": "BraTS 2020",
            },
            "seizure": {
                "task": "EEG seizure detection",
                "model": "Random Forest",
                "dataset": "CHB-MIT",
            },
            "motor": {
                "task": "Motor imagery BCI classification",
                "model": "CSP + LDA",
                "dataset": "PhysioNet EEGBCI",
            },
        },
    }


app.include_router(mri.router, prefix="/api/mri", tags=["MRI"])
app.include_router(seizure.router, prefix="/api/seizure", tags=["Seizure"])
app.include_router(motor.router, prefix="/api/motor", tags=["Motor"])
app.include_router(files.router, prefix="/api/files", tags=["Files"])