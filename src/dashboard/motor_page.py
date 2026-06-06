from __future__ import annotations

from pathlib import Path

import streamlit as st

from .utils import PROJECT_ROOT, ensure_demo_assets, display_path
from src.motor_imagery.inference import run_motor_dashboard_demo


def _show_image_or_info(
    image_path: Path,
    caption: str,
) -> None:
    """
    Safely show an image.
    If the image does not exist, show an info message instead of crashing.
    """

    if image_path.exists():
        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True,
        )
    else:
        st.info(f"Image not found yet: {image_path}")


def _render_motor_data_source_info() -> None:
    """
    Explain the current motor imagery data source and model status.
    """

    st.subheader("Data Source and Model")

    st.info(
        "Current data source: PhysioNet EEGBCI if `models/motor_csp_lda.pkl` "
        "was trained successfully. The CSP/LDA model is trained on real EEGBCI data. "
        "The dashboard prediction may use a synthetic demo epoch only for interface testing."
    )

    st.markdown(
        """
        **Dataset:** PhysioNet EEG Motor Movement/Imagery Dataset  
        **Task:** T1/T2 motor imagery classification  
        **Model:** CSP + LDA  
        **Frequency band:** 8–30 Hz  
        **Current status:** Real EEG training pipeline  
        """
    )


def _render_model_status(model_path: Path, cm_path: Path) -> None:
    """
    Show whether model and confusion matrix files exist.
    """

    st.subheader("Model Status")

    if model_path.exists():
        st.success("Motor imagery model found.")
        st.code(display_path(model_path))
    else:
        st.warning("Motor imagery model not found.")
        st.markdown(
            """
            Train the motor imagery model first:

            ```bash
            python -m scripts.train_motor_imagery
            ```
            """
        )

    st.divider()

    st.subheader("Expected Output Files")

    st.code(
        f"""
{display_path(model_path)}
{display_path(cm_path)}
        """.strip()
    )


def _render_motor_result(model_path: Path) -> None:
    """
    Run dashboard-safe motor inference if model exists.
    """

    st.subheader("Classification Result")

    if not model_path.exists():
        st.info("Motor inference result will appear after training the model.")
        return

    try:
        result = run_motor_dashboard_demo()

        st.metric("Prediction", result.get("prediction", "N/A"))

        confidence = result.get("confidence")

        if confidence is not None:
            st.metric("Confidence", f"{confidence:.2f}")
        else:
            st.metric("Confidence", "N/A")

        metrics = result.get("model_metrics", {})

        if metrics:
            st.divider()
            st.subheader("Saved Model Metrics")
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}")
            st.metric("F1-score", f"{metrics.get('f1_score', 0):.2f}")

            if "n_epochs" in metrics:
                st.metric("Epochs", metrics.get("n_epochs"))

            if "n_channels" in metrics:
                st.metric("Channels", metrics.get("n_channels"))

        else:
            st.info(
                "No saved metrics found in the model bundle. "
                "Retrain using `python -m scripts.train_motor_imagery` after updating train.py."
            )

    except Exception as e:
        st.warning("Motor imagery inference failed.")
        st.code(str(e))


def render_motor_page() -> None:
    """
    Render EEG motor imagery BCI page.

    This page should not crash even if:
    - models/motor_csp_lda.pkl does not exist
    - confusion matrix image does not exist
    - inference fails
    """

    ensure_demo_assets()

    st.title("EEG Motor Imagery BCI")
    st.caption("CSP + LDA left/right motor imagery classifier")

    st.markdown(
        """
        This module demonstrates an EEG-based motor imagery classifier using
        CSP spatial filtering and LDA classification.
        """
    )

    _render_motor_data_source_info()

    st.divider()

    model_path = PROJECT_ROOT / "models/motor_csp_lda.pkl"
    cm_path = PROJECT_ROOT / "results/motor/confusion_matrix_motor.png"
    topomap_path = PROJECT_ROOT / "results/motor/csp_patterns.png"

    col1, col2 = st.columns([1.4, 1.0])

    with col1:
        st.subheader("Motor Imagery Visualization")

        _show_image_or_info(
            image_path=topomap_path,
            caption="CSP topomap",
        )

        _show_image_or_info(
            image_path=cm_path,
            caption="Motor imagery confusion matrix",
        )

    with col2:
        _render_model_status(
            model_path=model_path,
            cm_path=cm_path,
        )

        st.divider()

        _render_motor_result(model_path=model_path)

    st.divider()

    st.subheader("Motor Imagery Pipeline")

    st.code(
        """
PhysioNet EEGBCI
↓
8–30 Hz band-pass filtering
↓
T1/T2 motor imagery epoch extraction
↓
CSP spatial filtering
↓
LDA classifier
↓
Motor imagery prediction
        """.strip()
    )

    st.divider()

    st.subheader("How to Generate Motor Results")

    st.markdown(
        """
        Run the motor imagery training script:

        ```bash
        python -m scripts.train_motor_imagery
        ```

        Expected outputs:

        ```text
        models/motor_csp_lda.pkl
        results/motor/confusion_matrix_motor.png
        results/motor/motor_metrics.json
        ```
        """
    )