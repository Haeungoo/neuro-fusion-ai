from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import mri, seizure, motor, files


app = FastAPI(
    title="NeuroFusion-AI API",
    description="FastAPI backend for NeuroFusion-AI neuroscience modules.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "project": "NeuroFusion-AI",
        "version": "0.2.0",
        "message": "NeuroFusion-AI FastAPI backend is running.",
        "modules": [
            "MRI Tumor Segmentation",
            "EEG Seizure Detection",
            "EEG Motor Imagery BCI",
        ],
    }


@app.get("/api/overview")
def overview() -> dict:
    return {
        "project": "NeuroFusion-AI",
        "description": (
            "Multimodal neuroscience AI system integrating MRI tumor segmentation, "
            "EEG seizure detection, and EEG motor imagery BCI."
        ),
        "version": "0.2.0",
        "backend": "FastAPI",
        "current_frontend": "Streamlit",
        "planned_frontend": "Next.js",
        "modules": {
            "mri": {
                "task": "Brain tumor segmentation",
                "model": "2D U-Net",
                "dataset": "BraTS 2020",
            },
            "seizure": {
                "task": "EEG seizure detection",
                "model": "Random Forest",
                "dataset": "Synthetic EEG + CHB-MIT",
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
app.include_router(motor.router, prefix="/api/motor", tags=["Motor Imagery"])
app.include_router(files.router, prefix="/api/files", tags=["Files"])