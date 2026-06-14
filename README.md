# NeuroFusion-AI

NeuroFusion-AI is a multimodal neuroscience AI web application that combines three brain-focused machine learning modules into one dashboard:

1. **MRI Tumor Segmentation**
2. **EEG Seizure Detection**
3. **EEG Motor Imagery BCI**

The project includes Python-based model training/evaluation scripts, a FastAPI backend, and a Next.js frontend dashboard.

---

## Project Overview

NeuroFusion-AI is designed as a portfolio-style research and engineering project for demonstrating how AI can be applied to neuroimaging and EEG analysis.

The current system supports:

* Brain MRI tumor segmentation with patient-level validation
* EEG seizure detection with CHB-MIT-style window-level evaluation
* EEG motor imagery classification using CSP + LDA
* FastAPI endpoints for serving result metadata and visualization files
* Next.js dashboard pages for module-level visualization

---

## Main Modules

### 1. MRI Tumor Segmentation

The MRI module performs 2D tumor segmentation on brain MRI slices.

Main features:

* UNet-style segmentation baseline
* Patient-level validation split
* Dice score, IoU, and pixel-level metrics
* Prediction mask visualization
* Overlay visualization
* FastAPI result endpoint
* Next.js MRI dashboard

Typical outputs:

```text
results/mri/mri_validation_metrics.json
results/mri/mri_validation_per_slice.csv
results/mri/mri_validation_per_case.csv
results/mri/mri_input_slice.png
results/mri/mri_ground_truth_mask.png
results/mri/mri_predicted_mask.png
results/mri/mri_probability_map.png
results/mri/mri_prediction_overlay.png
results/mri/mri_inference_comparison.png
```

---

### 2. EEG Seizure Detection

The seizure module performs binary seizure/non-seizure classification from EEG window-level features.

Main features:

* CHB-MIT EEG feature extraction
* Random Forest seizure classifier
* Window-level prediction CSV
* Sensitivity, specificity, precision, recall, F1 score
* Balanced accuracy
* False alarms per hour
* Confusion matrix visualization
* FastAPI result endpoint
* Next.js seizure dashboard

Main scripts:

```bash
python -m scripts.extract_seizure_chbmit_features
python -m scripts.train_random_forest_seizure
python -m scripts.evaluate_seizure
```

Typical outputs:

```text
results/seizure/chbmit_multi_file_features.npz
results/seizure/chbmit_multi_file_predictions.csv
results/seizure/seizure_metrics.json
results/seizure/seizure_metrics_per_file.csv
results/seizure/seizure_evaluation_confusion_matrix.png
models/seizure_random_forest.joblib
```

---

### 3. EEG Motor Imagery BCI

The motor imagery module classifies left-hand versus right-hand motor imagery EEG trials.

Main features:

* Synthetic EEG-like baseline
* PhysioNet EEGBCI real EEG pipeline
* CSP spatial filtering
* Linear Discriminant Analysis
* Subject-level performance search
* Accuracy, precision, recall, specificity, F1 score
* Confusion matrix visualization
* FastAPI result endpoint
* Next.js motor imagery dashboard

Label convention:

```text
0 = left_hand_imagery
1 = right_hand_imagery
```

---

## Tech Stack

### Machine Learning / Signal Processing

* Python
* NumPy
* pandas
* scikit-learn
* MNE
* matplotlib
* joblib

### Backend

* FastAPI
* Uvicorn
* Static file serving for result images

### Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS

---

## Project Structure

```text
neuro-fusion-ai/
├── backend/
│   ├── main.py
│   └── routers/
│       ├── mri.py
│       ├── seizure.py
│       └── motor.py
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── lib/
│   │   │   └── api.ts
│   │   ├── mri/
│   │   │   └── page.tsx
│   │   ├── seizure/
│   │   │   └── page.tsx
│   │   └── motor/
│   │       └── page.tsx
│   │
│   └── components/
│       ├── MriDashboard.tsx
│       ├── SeizureDashboard.tsx
│       └── MotorImageryDashboard.tsx
│
├── scripts/
│   ├── prepare_mri_brats_slices.py
│   ├── train_mri_unet2d.py
│   ├── evaluate_mri_validation.py
│   ├── test_mri_unet_forward.py
│   │
│   ├── extract_seizure_chbmit_features.py
│   ├── train_random_forest_seizure.py
│   ├── evaluate_seizure.py
│   │
│   ├── train_motor_imagery_csp_lda.py
│   ├── train_motor_imagery_physionet_csp_lda.py
│   └── train_motor_imagery_physionet_subject_search.py
│
├── src/
│   ├── common/
│   │   └── paths.py
│   │
│   ├── mri_segmentation/
│   │   ├── dataset.py
│   │   ├── metrics.py
│   │   ├── splits.py
│   │   └── train.py
│   │
│   ├── seizure_detection/
│   │   ├── metrics.py
│   │   └── train.py
│   │
│   └── motor_imagery/
│       ├── __init__.py
│       └── metrics.py
│
├── data/
├── results/
├── models/
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd neuro-fusion-ai
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, install the core packages:

```bash
pip install numpy pandas scikit-learn matplotlib joblib fastapi uvicorn mne
```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
```

---

## Running the Backend

From the project root:

```bash
cd ~/Downloads/neuro-fusion-ai
uvicorn backend.main:app --reload
```

Backend root:

```text
http://127.0.0.1:8000
```

FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

Static result files are served from:

```text
http://127.0.0.1:8000/media/results/
```

---

## Running the Frontend

Open a new terminal:

```bash
cd ~/Downloads/neuro-fusion-ai/frontend
npm run dev
```

Frontend:

```text
http://localhost:3000
```

Module pages:

```text
http://localhost:3000/mri
http://localhost:3000/seizure
http://localhost:3000/motor
```

---

## API Endpoints

### MRI

```text
GET /api/mri/status
```

### Seizure Detection

```text
GET /api/seizure/status
```

### Motor Imagery

```text
GET /api/motor/status
```

---

## Running MRI Module

### 1. Prepare MRI slices

```bash
python -m scripts.prepare_mri_brats_slices
```

### 2. Train MRI model

```bash
python -m scripts.train_mri_unet2d
```

### 3. Evaluate validation performance

```bash
python -m scripts.evaluate_mri_validation
```

### 4. Run inference visualization

```bash
python -m scripts.test_mri_unet_forward
```

Expected outputs:

```text
results/mri/mri_validation_metrics.json
results/mri/mri_validation_per_slice.csv
results/mri/mri_validation_per_case.csv
results/mri/mri_input_slice.png
results/mri/mri_ground_truth_mask.png
results/mri/mri_predicted_mask.png
results/mri/mri_probability_map.png
results/mri/mri_prediction_overlay.png
results/mri/mri_inference_comparison.png
```

---

## Running Seizure Detection Module

### 1. Extract CHB-MIT EEG features

```bash
python -m scripts.extract_seizure_chbmit_features
```

This creates:

```text
results/seizure/chbmit_multi_file_features.npz
```

### 2. Train Random Forest seizure classifier

```bash
python -m scripts.train_random_forest_seizure
```

This creates:

```text
results/seizure/chbmit_multi_file_predictions.csv
models/seizure_random_forest.joblib
```

### 3. Evaluate seizure detection

```bash
python -m scripts.evaluate_seizure
```

This creates:

```text
results/seizure/seizure_metrics.json
results/seizure/seizure_metrics_per_file.csv
results/seizure/seizure_evaluation_confusion_matrix.png
```

### Seizure metrics

The seizure module reports:

```text
accuracy
precision
recall
sensitivity
specificity
f1_score
balanced_accuracy
false_alarms_per_hour
```

### Seizure dashboard

```text
http://localhost:3000/seizure
```

---

## Running Motor Imagery Module

The motor imagery module supports three levels:

1. Synthetic baseline
2. PhysioNet single-subject training
3. PhysioNet subject search

---

### 1. Synthetic Motor Imagery Baseline

```bash
python -m scripts.train_motor_imagery_csp_lda
```

This generates:

```text
results/motor_imagery/motor_imagery_metrics.json
results/motor_imagery/motor_imagery_predictions.csv
results/motor_imagery/motor_imagery_confusion_matrix.png
models/motor_imagery_csp_lda.joblib
```

This version uses synthetic EEG-like data to validate the CSP + LDA pipeline.

---

### 2. PhysioNet EEGBCI Single-Subject Training

```bash
python -m scripts.train_motor_imagery_physionet_csp_lda
```

This trains CSP + LDA on real EEG motor imagery data using PhysioNet EEGBCI runs:

```text
4, 8, 12
```

These runs correspond to left-hand versus right-hand motor imagery.

Generated outputs:

```text
results/motor_imagery/motor_imagery_physionet_metrics.json
results/motor_imagery/motor_imagery_physionet_predictions.csv
results/motor_imagery/motor_imagery_physionet_confusion_matrix.png

results/motor_imagery/motor_imagery_metrics.json
results/motor_imagery/motor_imagery_predictions.csv
results/motor_imagery/motor_imagery_confusion_matrix.png

models/motor_imagery_physionet_csp_lda.joblib
models/motor_imagery_csp_lda.joblib
```

The generic output filenames are used by the dashboard.

---

### 3. PhysioNet Subject Search

```bash
python -m scripts.train_motor_imagery_physionet_subject_search
```

This evaluates multiple PhysioNet subjects and selects the best-performing subject for dashboard visualization.

Current subject search configuration:

```text
subjects = 1 to 10
runs = 4, 8, 12
band-pass filter = 8–30 Hz
epoch window = 0.5–3.5 seconds
CSP components = 6
classifier = Linear Discriminant Analysis
validation = 5-fold cross-validation
```

Generated outputs:

```text
results/motor_imagery/physionet_subject_comparison.csv
results/motor_imagery/physionet_subject_comparison_best.json
results/motor_imagery/physionet_subject_comparison_accuracy.png

results/motor_imagery/motor_imagery_metrics.json
results/motor_imagery/motor_imagery_predictions.csv
results/motor_imagery/motor_imagery_confusion_matrix.png

models/motor_imagery_csp_lda.joblib
models/motor_imagery_physionet_subject_search.joblib
```

### Motor imagery dashboard

```text
http://localhost:3000/motor
```

---

## Recommended Full Local Run

### 1. Generate model outputs

```bash
cd ~/Downloads/neuro-fusion-ai

python -m scripts.extract_seizure_chbmit_features
python -m scripts.train_random_forest_seizure
python -m scripts.evaluate_seizure

python -m scripts.train_motor_imagery_physionet_subject_search
```

MRI scripts can be run separately if MRI raw data and model training data are available:

```bash
python -m scripts.prepare_mri_brats_slices
python -m scripts.train_mri_unet2d
python -m scripts.evaluate_mri_validation
python -m scripts.test_mri_unet_forward
```

### 2. Run backend

```bash
uvicorn backend.main:app --reload
```

### 3. Run frontend

```bash
cd frontend
npm run dev
```

### 4. Open dashboards

```text
http://localhost:3000
http://localhost:3000/mri
http://localhost:3000/seizure
http://localhost:3000/motor
```

---

## Results and Model Files

The following directories are generated locally:

```text
results/
models/
data/
```

These are usually not committed to Git because they may contain large generated files or downloaded datasets.

Recommended `.gitignore` entries:

```text
results/
models/
data/
__pycache__/
.venv/
.next/
node_modules/
```

---

## Notes on Data

### MRI

MRI training requires prepared brain MRI slices and masks.

### Seizure Detection

The seizure module expects CHB-MIT style EEG data under:

```text
data/seizure_eeg/chb01
```

Typical required files:

```text
*.edf
chb01-summary.txt
```

### Motor Imagery

The motor imagery module can automatically download PhysioNet EEGBCI data through MNE when running:

```bash
python -m scripts.train_motor_imagery_physionet_subject_search
```

Downloaded files are stored under:

```text
data/motor_imagery/physionet
```

---

## Current Status

### Completed

* MRI segmentation dashboard structure
* MRI patient-level validation workflow
* Seizure detection evaluation metrics
* Seizure FastAPI integration
* Seizure dashboard
* Motor imagery synthetic CSP + LDA baseline
* Motor imagery PhysioNet EEGBCI training
* Motor imagery subject search
* Motor imagery FastAPI integration
* Motor imagery dashboard

### In Progress / Future Work

* Improve MRI segmentation with larger patient-level dataset
* Improve seizure demo visualizations with real EDF waveform plots
* Add cross-subject motor imagery evaluation
* Add CSP topomap visualization
* Add model comparison for motor imagery
* Add README figures and example screenshots
* Add automated tests for API endpoints

---

## Portfolio Summary

NeuroFusion-AI demonstrates an end-to-end neuroscience AI workflow:

```text
Raw neuro data
→ preprocessing
→ feature extraction or deep learning model
→ evaluation metrics
→ result visualization
→ backend API
→ frontend dashboard
```

The project combines medical imaging, EEG signal processing, machine learning, backend engineering, and frontend dashboard development in one integrated platform.
