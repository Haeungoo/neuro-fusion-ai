from __future__ import annotations

import streamlit as st


def render_about_page() -> None:
    st.title("About This Project")

    st.markdown(
        """
        ## NeuroFusion-AI

        **NeuroFusion-AI** is a UI-first multimodal neuroscience AI project.

        It combines three major neurotechnology directions:

        | Module | Data type | AI task |
        |---|---|---|
        | MRI Tumor Segmentation | Brain MRI | Medical image segmentation |
        | EEG Seizure Detection | Clinical EEG | Time-series classification |
        | EEG Motor Imagery BCI | EEG | Neural decoding |

        ## Why UI-first?

        UI-first development helps define the required input/output structure before the real models are completed.

        Each module has a placeholder inference function. Later, these functions can be replaced with real model code.

        ## Tech Stack

        - Python
        - Streamlit
        - NumPy
        - pandas
        - matplotlib
        - scikit-learn
        - MNE
        - PyTorch, later stage

        ## Suggested Next Steps

        1. Connect motor imagery module to CSP + LDA.
        2. Connect seizure module to CHB-MIT Random Forest classifier.
        3. Connect MRI module to 2D U-Net.
        4. Add final report and GitHub documentation.
        """
    )