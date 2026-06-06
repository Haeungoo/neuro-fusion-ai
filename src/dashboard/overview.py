from __future__ import annotations

from pathlib import Path
import json

import streamlit as st

from .utils import PROJECT_ROOT, ensure_demo_assets, display_path


def _file_status(path: Path, label: str) -> None:
    """
    Show whether a file exists.
    """

    if path.exists():
        st.success(f"{label} found")
    else:
        st.warning(f"{label} missing")

    st.code(display_path(path))


def _read_json(path: Path) -> dict:
    """
    Safely read JSON file.
    """

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def render_overview_page() -> None:
    """
    Render project overview page.
    """

    ensure_demo_assets()

    st.title("NeuroFusion-AI")
    st.caption("Multimodal neuroscience AI dashboard")

    st.markdown(
        """
        **NeuroFusion-AI** is a portfolio-style neuroscience AI project that combines
        three neuro/medical AI modules in one Streamlit dashboard:

        1. **MRI tumor segmentation**
        2. **EEG seizure detection**
        3. **EEG motor imagery BCI classification**

        The goal of this project is to demonstrate practical AI workflows for
        brain MRI and EEG signals, including preprocessing, model training,
        inference, visualization, and result interpretation.
        """
    )

    st.divider()

    st.subheader("Project Modules")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### MRI Tumor Segmentation")
        st.write("**Dataset:** BraTS 2020 / synthetic fallback")
        st.write("**Model:** 2D U-Net")
        st.write("**Output:** MRI tumor mask overlay")
        st.write("**Status:** Real-data pipeline in progress")

    with col2:
        st.markdown("### EEG Seizure Detection")
        st.write("**Dataset:** Synthetic EEG + CHB-MIT")
        st.write("**Model:** Random Forest")
        st.write("**Output:** Seizure probability timeline")
        st.write("**Status:** Synthetic, one-file, and multi-file modes")

    with col3:
        st.markdown("### EEG Motor Imagery BCI")
        st.write("**Dataset:** PhysioNet EEGBCI")
        st.write("**Model:** CSP + LDA")
        st.write("**Output:** Motor imagery classification")
        st.write("**Status:** Real EEG prototype")

    st.divider()

    st.subheader("Current File Status")

    tab_mri, tab_seizure, tab_motor = st.tabs(
        ["MRI", "Seizure", "Motor Imagery"]
    )

    with tab_mri:
        _file_status(
            PROJECT_ROOT / "models/mri_unet.pt",
            "MRI U-Net model",
        )

        _file_status(
            PROJECT_ROOT / "results/mri/mri_prediction_overlay.png",
            "MRI prediction overlay",
        )

        _file_status(
            PROJECT_ROOT / "results/mri/mri_training_metrics.json",
            "MRI training metrics",
        )

        _file_status(
            PROJECT_ROOT / "results/mri/mri_inference_metrics.json",
            "MRI inference metrics",
        )

    with tab_seizure:
        _file_status(
            PROJECT_ROOT / "models/seizure_rf.pkl",
            "Synthetic seizure model",
        )

        _file_status(
            PROJECT_ROOT / "models/seizure_rf_chbmit_one_file.pkl",
            "CHB-MIT one-file seizure model",
        )

        _file_status(
            PROJECT_ROOT / "models/seizure_rf_chbmit_multi_file.pkl",
            "CHB-MIT multi-file seizure model",
        )

        _file_status(
            PROJECT_ROOT / "results/seizure/chbmit_multi_file_probability_timeline.png",
            "CHB-MIT multi-file probability timeline",
        )

    with tab_motor:
        _file_status(
            PROJECT_ROOT / "models/motor_csp_lda.pkl",
            "Motor imagery CSP+LDA model",
        )

        _file_status(
            PROJECT_ROOT / "results/motor/confusion_matrix_motor.png",
            "Motor imagery confusion matrix",
        )

        _file_status(
            PROJECT_ROOT / "results/motor/csp_patterns.png",
            "CSP spatial patterns",
        )

        _file_status(
            PROJECT_ROOT / "results/motor/motor_metrics.json",
            "Motor training metrics",
        )

    st.divider()

    st.subheader("Key Metrics Summary")

    mri_metrics = _read_json(
        PROJECT_ROOT / "results/mri/mri_inference_metrics.json"
    )

    motor_metrics = _read_json(
        PROJECT_ROOT / "results/motor/motor_metrics.json"
    )

    seizure_metrics = _read_json(
        PROJECT_ROOT / "results/seizure/seizure_metrics.json"
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### MRI")
        dice_score = mri_metrics.get("dice_score")
        if dice_score is not None:
            st.metric("Dice score", f"{dice_score:.4f}")
        else:
            st.info("Dice score not available yet.")

    with col_b:
        st.markdown("### Seizure")
        f1 = seizure_metrics.get("f1_score")
        sensitivity = seizure_metrics.get("sensitivity")
        if f1 is not None:
            st.metric("F1-score", f"{f1:.4f}")
        else:
            st.info("F1-score not available yet.")

        if sensitivity is not None:
            st.metric("Sensitivity", f"{sensitivity:.4f}")

    with col_c:
        st.markdown("### Motor")
        accuracy = motor_metrics.get("accuracy")
        f1_score = motor_metrics.get("f1_score")

        if accuracy is not None:
            st.metric("Accuracy", f"{accuracy:.4f}")
        else:
            st.info("Accuracy not available yet.")

        if f1_score is not None:
            st.metric("F1-score", f"{f1_score:.4f}")

    st.divider()

    st.subheader("System Pipeline")

    st.code(
        """
MRI branch:
BraTS MRI slice → 2D U-Net → Tumor mask → Overlay + Dice score

Seizure branch:
EEG signal → Sliding windows → Feature extraction → Random Forest → Probability timeline

Motor imagery branch:
PhysioNet EEGBCI → Band-pass filtering → Epoching → CSP → LDA → Motor imagery class
        """.strip()
    )

    st.divider()

    st.subheader("Current Limitations")

    st.markdown(
        """
        - MRI model quality depends on the number of BraTS slices used for training.
        - CHB-MIT seizure detection is still a prototype and should not be interpreted as clinical software.
        - Motor imagery performance may drop when multiple subjects are mixed together.
        - The dashboard is designed for educational and portfolio demonstration purposes.
        """
    )

    st.subheader("Next Development Steps")

    st.markdown(
        """
        1. Improve MRI segmentation with more BraTS cases and Dice/IoU evaluation.
        2. Expand seizure detection to patient-independent CHB-MIT evaluation.
        3. Add subject-level benchmarking for motor imagery.
        4. Add upload functionality for EEG EDF and MRI NIfTI files.
        5. Prepare screenshots and GitHub documentation.
        """
    )