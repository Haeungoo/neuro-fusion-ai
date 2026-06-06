from __future__ import annotations

import streamlit as st

from src.dashboard.overview import render_overview_page
from src.dashboard.mri_page import render_mri_page
from src.dashboard.seizure_page import render_seizure_page
from src.dashboard.motor_page import render_motor_page
from src.dashboard.about_page import render_about_page


st.set_page_config(
    page_title="NeuroFusion-AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.sidebar.title("🧠 NeuroFusion-AI")
    st.sidebar.caption("Multimodal Brain AI Dashboard")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "MRI Tumor Segmentation",
            "EEG Seizure Detection",
            "EEG Motor Imagery BCI",
            "About",
        ],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Project Info")
    st.sidebar.write(
        "UI-first prototype integrating MRI segmentation, EEG seizure detection, and EEG motor imagery decoding."
    )

    st.sidebar.divider()
    st.sidebar.caption("Developed for Neuroscience + CS portfolio")

    if page == "Overview":
        render_overview_page()
    elif page == "MRI Tumor Segmentation":
        render_mri_page()
    elif page == "EEG Seizure Detection":
        render_seizure_page()
    elif page == "EEG Motor Imagery BCI":
        render_motor_page()
    elif page == "About":
        render_about_page()


if __name__ == "__main__":
    main()