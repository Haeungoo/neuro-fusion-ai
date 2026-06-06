from __future__ import annotations

from pathlib import Path
import json

import streamlit as st

from .utils import PROJECT_ROOT, ensure_demo_assets, display_path


def _show_image_card(image_path: Path, title: str, caption: str = "") -> None:
    """
    Show one image as a compact card.
    """
    st.markdown(f"**{title}**")

    if image_path.exists():
        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True,
        )
    else:
        st.info(f"Image not found: {display_path(image_path)}")

def _read_json_safe(path: Path) -> dict:
    """
    Safely read a JSON file.
    """
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _show_image_or_info(
    image_path: Path,
    caption: str,
) -> None:
    """
    Safely show an image in Streamlit.
    If image_path does not exist, show an info message instead of crashing.
    """

    if image_path.exists():
        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True,
        )
    else:
        st.info(f"Image not found yet: {image_path}")


def _render_mri_data_source_info() -> None:
    """
    Explain current MRI data source and model limitation.
    """

    st.subheader("Data Source and Model")

    st.warning(
        "Current data source: synthetic MRI-like image. "
        "If `models/mri_unet.pt` does not exist, the page uses a randomly initialized U-Net. "
        "Therefore, the current mask is only a structural forward-pass demo, not a clinically meaningful tumor segmentation result."
    )

    st.markdown(
        """
        **Current data:** Synthetic MRI-like 2D image  
        **Model:** 2D U-Net  
        **Current status:** Structural forward-pass demo  
        **Clinical meaning:** Not clinically meaningful unless trained on real MRI/mask pairs  
        **Future dataset:** BraTS brain tumor MRI dataset  
        """
    )


def _render_model_status(model_path: Path, output_path: Path) -> None:
    """
    Show whether trained MRI model and output image exist.
    """

    st.subheader("Model Status")

    if model_path.exists():
        st.success("Trained MRI U-Net model found.")
        st.code(display_path(model_path))
    else:
        st.warning("Trained MRI U-Net model not found.")
        st.write(
            "This is okay at the prototype stage. "
            "The app can still run a forward-pass demo using a randomly initialized U-Net."
        )

    st.divider()

    st.subheader("Expected Output Files")

    st.code(
    f"""
{display_path(model_path)}
{display_path(output_path)}
{display_path(PROJECT_ROOT / "results/mri/mri_input_slice.png")}
{display_path(PROJECT_ROOT / "results/mri/mri_ground_truth_mask.png")}
{display_path(PROJECT_ROOT / "results/mri/mri_predicted_mask.png")}
{display_path(PROJECT_ROOT / "results/mri/mri_prediction_overlay.png")}
    """.strip()
)

def _render_training_metrics() -> None:
    """
    Show MRI U-Net training metrics if the metrics JSON exists.
    """

    st.subheader("Training Metrics")

    metrics_path = PROJECT_ROOT / "results/mri/mri_training_metrics.json"
    curve_path = PROJECT_ROOT / "results/mri/training_curve_mri_unet.png"

    if not metrics_path.exists():
        st.info(
            "Training metrics not found yet. "
            "Run `python -m scripts.train_mri_unet2d`."
        )
        return

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        best_val_loss = metrics.get("best_val_loss")
        epochs = metrics.get("epochs")
        num_samples = metrics.get("num_samples")
        train_samples = metrics.get("train_samples")
        val_samples = metrics.get("val_samples")
        batch_size = metrics.get("batch_size")
        learning_rate = metrics.get("learning_rate")

        if best_val_loss is not None:
            st.metric("Best validation loss", f"{best_val_loss:.4f}")

        if epochs is not None:
            st.metric("Epochs", epochs)

        if num_samples is not None:
            st.metric("Total slices", num_samples)

        if train_samples is not None:
            st.metric("Train slices", train_samples)

        if val_samples is not None:
            st.metric("Validation slices", val_samples)

        if batch_size is not None:
            st.metric("Batch size", batch_size)

        if learning_rate is not None:
            st.metric("Learning rate", learning_rate)

        if curve_path.exists():
            st.image(
                str(curve_path),
                caption="MRI U-Net training curve",
                use_container_width=True,
            )
        else:
            st.info("Training curve image not found yet.")

    except Exception as e:
        st.warning("Could not read MRI training metrics.")
        st.code(str(e))


def _render_inference_metrics() -> None:
    """
    Show MRI inference metrics such as Dice score.
    """

    st.subheader("Inference Metrics")

    metrics_path = PROJECT_ROOT / "results/mri/mri_inference_metrics.json"

    if not metrics_path.exists():
        st.info(
            "Inference metrics not found yet. "
            "Run `python -m scripts.test_mri_unet_forward`."
        )
        return

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        sample_name = metrics.get("sample_name", "N/A")
        dice_score = metrics.get("dice_score")
        threshold = metrics.get("threshold")
        prediction_pixels = metrics.get("prediction_pixels")
        ground_truth_pixels = metrics.get("ground_truth_pixels")

        st.write("Sample:", sample_name)

        if dice_score is not None:
            st.metric("Dice score", f"{dice_score:.4f}")

        if threshold is not None:
            st.metric("Threshold", threshold)

        if prediction_pixels is not None:
            st.metric("Predicted pixels", prediction_pixels)

        if ground_truth_pixels is not None:
            st.metric("Ground-truth pixels", ground_truth_pixels)

        st.caption(
            "Dice score measures overlap between the predicted tumor mask and the ground-truth mask. "
            "A higher value indicates better segmentation overlap."
        )

    except Exception as e:
        st.warning("Could not read MRI inference metrics.")
        st.code(str(e))


def _render_mri_inference_result(output_path: Path) -> None:
    """
    Show MRI input, ground truth mask, predicted mask, and overlay
    in a compact 2x2 grid layout.
    """

    st.subheader("MRI Segmentation Result")

    st.markdown(
        """
        This section shows the full segmentation flow:

        **Input MRI slice → Ground-truth mask → Predicted mask → Prediction overlay**
        """
    )

    input_path = PROJECT_ROOT / "results/mri/mri_input_slice.png"
    gt_path = PROJECT_ROOT / "results/mri/mri_ground_truth_mask.png"
    pred_path = PROJECT_ROOT / "results/mri/mri_predicted_mask.png"
    overlay_path = PROJECT_ROOT / "results/mri/mri_prediction_overlay.png"

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        _show_image_card(
            image_path=input_path,
            title="1. Input MRI slice",
            caption="Original BraTS FLAIR MRI slice",
        )

    with row1_col2:
        _show_image_card(
            image_path=gt_path,
            title="2. Ground-truth tumor mask",
            caption="Tumor mask from BraTS segmentation label",
        )

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        _show_image_card(
            image_path=pred_path,
            title="3. Predicted tumor mask",
            caption="Binary mask predicted by 2D U-Net",
        )

    with row2_col2:
        _show_image_card(
            image_path=overlay_path,
            title="4. MRI + predicted overlay",
            caption="Predicted tumor mask overlaid on MRI",
        )

    st.info(
        "The ground-truth mask comes from the BraTS segmentation label. "
        "The predicted mask is generated by the trained 2D U-Net."
    )


def _render_mri_metric_bar() -> None:
    """
    Show compact MRI metrics at the top of the page.
    """

    inference_metrics_path = PROJECT_ROOT / "results/mri/mri_inference_metrics.json"
    training_metrics_path = PROJECT_ROOT / "results/mri/mri_training_metrics.json"
    model_path = PROJECT_ROOT / "models/mri_unet.pt"

    inference_metrics = _read_json_safe(inference_metrics_path)
    training_metrics = _read_json_safe(training_metrics_path)

    dice_score = inference_metrics.get("dice_score")
    pred_pixels = inference_metrics.get("prediction_pixels")
    gt_pixels = inference_metrics.get("ground_truth_pixels")
    best_val_loss = training_metrics.get("best_val_loss")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if dice_score is not None:
            st.metric("Dice score", f"{dice_score:.4f}")
        else:
            st.metric("Dice score", "N/A")

    with col2:
        if pred_pixels is not None:
            st.metric("Predicted pixels", f"{pred_pixels:,}")
        else:
            st.metric("Predicted pixels", "N/A")

    with col3:
        if gt_pixels is not None:
            st.metric("GT pixels", f"{gt_pixels:,}")
        else:
            st.metric("GT pixels", "N/A")

    with col4:
        if best_val_loss is not None:
            st.metric("Best val loss", f"{best_val_loss:.4f}")
        else:
            st.metric("Best val loss", "N/A")

    with col5:
        if model_path.exists():
            st.metric("Model", "Found")
        else:
            st.metric("Model", "Missing")
            

def render_mri_page() -> None:
    """
    Render MRI tumor segmentation page.

    This page is designed not to crash even if:
    - models/mri_unet.pt does not exist
    - results/mri/mri_prediction_overlay.png does not exist
    - MRI inference fails
    """

    ensure_demo_assets()

    st.title("MRI Tumor Segmentation")
    st.caption("2D U-Net brain tumor segmentation module")

    st.markdown(
        """
        This module demonstrates a 2D U-Net MRI tumor segmentation pipeline.
        At the current stage, it is mainly used to verify model architecture,
        inference flow, and visualization.
        """
    )

    _render_mri_data_source_info()

    st.divider()

    _render_mri_metric_bar()

    st.divider()

    model_path = PROJECT_ROOT / "models/mri_unet.pt"
    output_path = PROJECT_ROOT / "results/mri/mri_prediction_overlay.png"

    col1, col2 = st.columns([1.4, 1.0])

    with col1:
        _render_mri_inference_result(output_path=output_path)

    with col2:
        _render_model_status(
            model_path=model_path,
            output_path=output_path,
        )

        st.divider()

        _render_training_metrics()

        st.divider()

        _render_inference_metrics()

    st.subheader("MRI Segmentation Pipeline")

    st.code(
        """
Synthetic or real MRI 2D slice
↓
Intensity normalization
↓
2D U-Net
↓
Sigmoid probability mask
↓
Thresholding
↓
Overlay visualization
        """.strip()
    )

    st.divider()

    st.subheader("How to Generate MRI Demo Output")

    st.markdown(
        """
        Run the MRI forward-pass test:

        ```bash
        python -m scripts.test_mri_unet_forward
        ```

        Expected output:

        ```text
        results/mri/mri_prediction_overlay.png
        ```

        Later, after preparing real MRI image/mask pairs, run:

        ```bash
        python -m scripts.train_mri_unet2d
        ```
        """
    )

    st.divider()

    st.subheader("Future MRI Expansion")

    st.markdown(
        """
        Planned BraTS integration:

        1. Load BraTS NIfTI files  
        2. Extract 2D MRI slices  
        3. Match MRI slices with tumor masks  
        4. Train 2D U-Net on real image/mask pairs  
        5. Evaluate Dice score and IoU  
        6. Display real tumor overlay results in the dashboard  
        """
    )