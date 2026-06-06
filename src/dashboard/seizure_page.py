from __future__ import annotations

from pathlib import Path

import streamlit as st

from .utils import PROJECT_ROOT, ensure_demo_assets, display_path


def _show_image_or_warning(
    image_path: Path,
    caption: str,
    fallback_path: Path | None = None,
) -> None:
    """
    Safely show an image in Streamlit.

    If image_path does not exist, fallback_path is used.
    If neither exists, a warning message is shown instead of crashing the page.
    """

    if image_path.exists():
        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True,
        )
    elif fallback_path is not None and fallback_path.exists():
        st.image(
            str(fallback_path),
            caption=f"Fallback: {caption}",
            use_container_width=True,
        )
    else:
        st.warning(f"Image not found: {display_path(image_path)}")


def _render_selected_source_info(data_source: str) -> None:
    """
    Render explanation for the selected seizure data source.
    """

    st.subheader("Selected Data Source")

    if data_source == "CHB-MIT chb01_03 real EEG":
        st.success("Selected source: CHB-MIT chb01_03 real EEG")

        st.markdown(
            """
            **Dataset type:** Real scalp EEG  
            **Dataset:** CHB-MIT Scalp EEG Database  
            **File:** `chb01_03.edf`  
            **Known seizure interval:** 2996–3036 sec  
            **Model:** Random Forest classifier  
            **Window size:** 5 sec  
            **Step size:** 2.5 sec  
            **Current status:** One-file real EEG prototype  
            """
        )

    elif data_source == "CHB-MIT multi-file real EEG":
        st.success("Selected source: CHB-MIT multi-file real EEG")

        st.markdown(
            """
            **Dataset type:** Real scalp EEG  
            **Dataset:** CHB-MIT Scalp EEG Database  
            **Patient:** `chb01`  
            **Files:** multiple seizure-containing EDF files  
            **Model:** Random Forest classifier  
            **Window size:** 5 sec  
            **Step size:** 2.5 sec  
            **Current status:** Multi-file subject-level prototype  
            """
        )

    else:
        st.success("Selected source: Synthetic demo EEG")

        st.markdown(
            """
            **Dataset type:** Synthetic EEG-like signal  
            **Purpose:** Pipeline testing before real EEG integration  
            **Model:** Random Forest classifier  
            **Window size:** 5 sec  
            **Step size:** 2.5 sec  
            **Current status:** Demo pipeline test  
            """
        )


def _get_paths_for_source(data_source: str) -> dict:
    """
    Return model path, result image paths, and commands for each seizure source.
    """

    if data_source == "CHB-MIT chb01_03 real EEG":
        return {
            "model_path": PROJECT_ROOT / "models/seizure_rf_chbmit_one_file.pkl",
            "waveform_path": PROJECT_ROOT / "results/seizure/chbmit_chb01_03_waveform.png",
            "timeline_path": PROJECT_ROOT / "results/seizure/chbmit_chb01_03_probability_timeline.png",
            "confusion_matrix_path": PROJECT_ROOT / "results/seizure/chbmit_chb01_03_confusion_matrix.png",
            "fallback_waveform": PROJECT_ROOT / "results/seizure/eeg_waveform_input.png",
            "fallback_timeline": PROJECT_ROOT / "results/seizure/seizure_probability_timeline.png",
            "fallback_confusion_matrix": PROJECT_ROOT / "results/seizure/confusion_matrix_seizure.png",
            "train_command": "python -m scripts.train_seizure_chbmit_one_file",
            "test_command": "python -m scripts.test_seizure_chbmit_one_file",
            "copy_commands": [
                "cp results/seizure/eeg_waveform_input.png results/seizure/chbmit_chb01_03_waveform.png",
                "cp results/seizure/seizure_probability_timeline.png results/seizure/chbmit_chb01_03_probability_timeline.png",
                "cp results/seizure/confusion_matrix_seizure.png results/seizure/chbmit_chb01_03_confusion_matrix.png",
            ],
        }

    if data_source == "CHB-MIT multi-file real EEG":
        return {
            "model_path": PROJECT_ROOT / "models/seizure_rf_chbmit_multi_file.pkl",
            "waveform_path": PROJECT_ROOT / "results/seizure/chbmit_multi_file_waveform.png",
            "timeline_path": PROJECT_ROOT / "results/seizure/chbmit_multi_file_probability_timeline.png",
            "confusion_matrix_path": PROJECT_ROOT / "results/seizure/chbmit_multi_file_confusion_matrix.png",
            "fallback_waveform": PROJECT_ROOT / "results/seizure/eeg_waveform_input.png",
            "fallback_timeline": PROJECT_ROOT / "results/seizure/seizure_probability_timeline.png",
            "fallback_confusion_matrix": PROJECT_ROOT / "results/seizure/confusion_matrix_seizure.png",
            "train_command": "python -m scripts.train_seizure_chbmit_multi_file",
            "test_command": "python -m scripts.test_seizure_chbmit_multi_file",
            "copy_commands": [
                "ls results/seizure/chbmit_multi_file_confusion_matrix.png",
                "ls results/seizure/chbmit_multi_file_waveform.png",
                "ls results/seizure/chbmit_multi_file_probability_timeline.png",
            ],
        }

    return {
        "model_path": PROJECT_ROOT / "models/seizure_rf.pkl",
        "waveform_path": PROJECT_ROOT / "results/seizure/synthetic_waveform.png",
        "timeline_path": PROJECT_ROOT / "results/seizure/synthetic_probability_timeline.png",
        "confusion_matrix_path": PROJECT_ROOT / "results/seizure/synthetic_confusion_matrix.png",
        "fallback_waveform": PROJECT_ROOT / "results/seizure/eeg_waveform_input.png",
        "fallback_timeline": PROJECT_ROOT / "results/seizure/seizure_probability_timeline.png",
        "fallback_confusion_matrix": PROJECT_ROOT / "results/seizure/confusion_matrix_seizure.png",
        "train_command": "python -m scripts.train_seizure_demo",
        "test_command": "python -m scripts.test_seizure_inference",
        "copy_commands": [
            "cp results/seizure/eeg_waveform_input.png results/seizure/synthetic_waveform.png",
            "cp results/seizure/seizure_probability_timeline.png results/seizure/synthetic_probability_timeline.png",
            "cp results/seizure/confusion_matrix_seizure.png results/seizure/synthetic_confusion_matrix.png",
        ],
    }


def _render_model_status(paths: dict) -> None:
    """
    Render model status and expected output file paths.
    """

    st.subheader("Model Status")

    model_path = paths["model_path"]

    if model_path.exists():
        st.success("Seizure model found.")
        st.code(display_path(model_path), language="text")
    else:
        st.warning("Seizure model not found.")
        st.markdown("Train the selected model first:")
        st.code(paths["train_command"], language="bash")

    st.divider()

    st.subheader("Expected Output Files")

    st.code(
        f"""
{display_path(paths["waveform_path"])}
{display_path(paths["timeline_path"])}
{display_path(paths["confusion_matrix_path"])}
        """.strip(),
        language="text",
    )


def _render_generation_commands(paths: dict) -> None:
    """
    Render commands needed to generate and preserve result files.

    This version uses st.code() separately to avoid copy_commands
    being displayed as a comma-separated Python list.
    """

    st.subheader("How to Generate These Result Files")

    st.markdown("**1. Run training**")
    st.code(paths["train_command"], language="bash")

    st.markdown("**2. Run inference**")
    st.code(paths["test_command"], language="bash")

    copy_commands = paths.get("copy_commands", [])

    if copy_commands:
        st.markdown("**3. Preserve or verify output files**")

        if isinstance(copy_commands, list):
            commands_text = "\n".join(str(cmd) for cmd in copy_commands)
        else:
            commands_text = str(copy_commands)

        st.code(commands_text, language="bash")


def render_seizure_page() -> None:
    """
    Render EEG seizure detection page.

    This page compares:
    1. Synthetic demo EEG
    2. CHB-MIT chb01_03 real EEG one-file prototype
    3. CHB-MIT multi-file real EEG prototype

    The page is designed to avoid Streamlit crashes if files are missing.
    """

    ensure_demo_assets()

    st.title("EEG Seizure Detection")
    st.caption("Random Forest seizure classifier using EEG window features")

    st.markdown(
        """
        This module demonstrates an EEG seizure detection pipeline using
        sliding windows, EEG feature extraction, and a Random Forest classifier.
        """
    )

    data_source = st.radio(
        "Select seizure data source",
        [
            "Synthetic demo EEG",
            "CHB-MIT chb01_03 real EEG",
            "CHB-MIT multi-file real EEG",
        ],
        horizontal=True,
    )

    _render_selected_source_info(data_source)

    paths = _get_paths_for_source(data_source)

    st.divider()

    col1, col2 = st.columns([1.5, 1.0])

    with col1:
        st.subheader("EEG Signal and Seizure Probability")

        _show_image_or_warning(
            image_path=paths["waveform_path"],
            caption="EEG waveform",
            fallback_path=paths["fallback_waveform"],
        )

        _show_image_or_warning(
            image_path=paths["timeline_path"],
            caption="Seizure probability timeline",
            fallback_path=paths["fallback_timeline"],
        )

        _show_image_or_warning(
            image_path=paths["confusion_matrix_path"],
            caption="Seizure classifier confusion matrix",
            fallback_path=paths["fallback_confusion_matrix"],
        )

    with col2:
        _render_model_status(paths)

    st.divider()

    st.subheader("Seizure Detection Pipeline")

    st.code(
        """
EEG signal
↓
5-second sliding windows
↓
Feature extraction
- Bandpower
- Entropy
- Line length
- Hjorth parameters
↓
Random Forest classifier
↓
Seizure probability timeline
        """.strip(),
        language="text",
    )

    st.divider()

    _render_generation_commands(paths)