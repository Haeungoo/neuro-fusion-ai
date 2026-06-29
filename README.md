# NeuroFusion-AI

**NeuroFusion-AI** is a full-stack neuroscience AI dashboard that combines MRI and EEG machine learning workflows in one web application.

The project demonstrates how medical imaging, EEG signal analysis, machine learning, backend APIs, and modern frontend dashboards can be integrated into a single portfolio system.

---

## Screenshots

### Overview Dashboard

![Overview Dashboard](screenshots/final/overview.png)

The overview page summarizes the current status and key metrics from all three AI modules.

### MRI Tumor Segmentation

![MRI Tumor Segmentation](screenshots/final/mri_page.png)
![MRI Tumor Segmentation](screenshots/final/mri_page_2.png)
![MRI Tumor Segmentation](screenshots/final/mri_page_3.png)

- FLAIR-based MRI tumor segmentation workflow
- U-Net-style segmentation model
- Predicted tumor mask visualization
- Ground-truth mask comparison
- Dice and IoU evaluation
- Best and worst validation case review
- Overlay visualization for easier interpretation

### EEG Seizure Detection

![EEG Seizure Detection](screenshots/final/seizure_page.png)
![EEG Seizure Detection](screenshots/final/seizure_page_2.png)
![EEG Seizure Detection](screenshots/final/seizure_page_3.png)

- EEG signal classification workflow
- Seizure vs non-seizure prediction
- EEG waveform visualization
- Probability timeline visualization
- Confusion matrix and classification metrics
- Prediction CSV output

### EEG Motor Imagery BCI

![EEG Motor Imagery BCI](screenshots/final/motor_page.png)
![EEG Motor Imagery BCI](screenshots/final/motor_page_2.png)
![EEG Motor Imagery BCI](screenshots/final/motor_page_3.png)

- PhysioNet EEGBCI motor imagery workflow
- Left-hand vs. right-hand imagery classification
- EEG preprocessing and epoch extraction
- CSP feature extraction
- LDA classification
- Stratified cross-validation
- Subject comparison
- CSP topomap visualization

---

### Dataset & Training Pipeline

#### 1. MRI Tumor Segmentation

The MRI module uses a FLAIR-based brain tumor segmentation workflow inspired by BraTS-style MRI data.

- **Input data**: *FLAIR MRI slices*
- **Task**: *Pixel-level tumor mask segmentation*
- **Model**: *U-Net-style segmentation model*
- **Training objective**: *Predict tumor segmentation masks from MRI slices*
- **Output**: *Predicted tumor mask and overlay visualization*
- **Evaluation**: *Dice score and Intersection over Union*
- **Dashboard outputs**: *Input MRI, ground-truth mask, predicted mask, overlay image, and best/worst validation case review*

The model predicts tumor regions from FLAIR MRI slices. The predicted mask is compared with the ground-truth segmentation mask using Dice and IoU to measure how well the predicted tumor region overlaps with the true tumor region.

#### 2. EEG Seizure Detection

The seizure detection module uses EEG time-series data to classify seizure-related patterns and visualize model outputs.

- **Input data**: *EEG signal windows*
- **Task**: *Seizure vs. non-seizure classification*
- **Model type**: *Feature-based machine learning classifier*
- **Training approach**: *Extract EEG signal features from time-series windows and train a classifier to distinguish seizure-related and non-seizure patterns*
- **Output**: *Prediction results, EEG waveform visualization, probability timeline, and confusion matrix*
- **Evaluation**: *Accuracy, precision, recall, specificity, F1 score, and confusion matrix*

This module focuses on transforming EEG time-series signals into interpretable classification outputs. The dashboard visualizes both the EEG waveform and the model’s seizure probability timeline to make the results easier to review.

#### 3. EEG Motor Imagery BCI

The motor imagery module uses PhysioNet EEGBCI data for left-hand vs. right-hand motor imagery classification.

- **Input data**: *EEG motor imagery trials from PhysioNet EEGBCI*
- **Task**: *Left-hand vs. right-hand motor imagery classification*
- **Pipeline**: *EEG preprocessing, epoch extraction, CSP feature extraction, and LDA classification*
- **Model**: *Common Spatial Patterns + Linear Discriminant Analysis*
- **Evaluation**: *Stratified cross-validation, accuracy, precision, recall, specificity, F1 score, and confusion matrix*
- **Dashboard outputs**: *Subject comparison, confusion matrix, prediction CSV, and CSP topomap visualization*

This module demonstrates a classical brain-computer interface workflow. CSP is used to extract spatial EEG patterns that help distinguish left-hand and right-hand motor imagery, and LDA is used as the final classifier.

---

## Tech Stack

### Backend

* Python
* FastAPI
* NumPy
* Pandas
* Scikit-learn
* PyTorch
* MNE
* Matplotlib
* Joblib

### Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS
* lucide-react

### Data / Output

* JSON metrics
* CSV prediction summaries
* PNG visualizations
* Saved model files
* Local dashboard API outputs

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
│   │   ├── mri/page.tsx
│   │   ├── seizure/page.tsx
│   │   └── motor/page.tsx
│   │   └── lib/
│   │       └── api.ts
│   └── components/
│
├── scripts/
│   ├── evaluate_mri_validation.py
│   ├── generate_seizure_waveform_timeline.py
│   ├── generate_seizure_predictions_csv.py
│   └── train_motor_imagery_physionet_subject_search.py
│
├── src/
│   ├── mri_segmentation/
│   ├── seizure_detection/
│   └── motor_imagery/
│
├── results/
├── models/
├── data/
├── screenshots/final/
└── README.md
```

Note: Large datasets, trained model checkpoints, and generated result files are not included in this repository.

---

## How to Run

### 1. Clone the repository
``` 
git clone https://github.com/YOUR_USERNAME/neuro-fusion-ai.git
cd neuro-fusion-ai
```

### 2. Create and activate a Python environment
```
source .venv/bin/activate
python -m venv .venv
```

### 3. Install Python dependencies
```
pip install -r requirements.txt
```

### 4. Run the FastAPI backend
```
uvicorn backend.main:app --reload
```

The backend will run at:
```
http://127.0.0.1:8000
```

API documentation:
```
http://127.0.0.1:8000/docs
```

### 5. Run the Next.js frontend
Open a second terminal:
```
cd frontend
npm install
npm run dev
```

The frontend will run at:
```
http://localhost:3000
```
---

## Dashboard Pages

```text
Overview:
http://localhost:3000

MRI Tumor Segmentation:
http://localhost:3000/mri

EEG Seizure Detection:
http://localhost:3000/seizure

EEG Motor Imagery BCI:
http://localhost:3000/motor
```

---

## API Endpoints

```text
GET /api/mri/status
GET /api/seizure/status
GET /api/motor/status
```

The frontend reads these endpoints to display live model metrics and result images.

---

## Generate Result Files

### MRI validation outputs

```bash
python -m scripts.evaluate_mri_validation
```

### Seizure waveform and timeline

```bash
python -m scripts.generate_seizure_waveform_timeline
python -m scripts.generate_seizure_predictions_csv
```

### Motor imagery results

```bash
python -m scripts.train_motor_imagery_physionet_subject_search
```

---

### What I Learned

Through this project, I practiced:

- MRI modality interpretation
- FLAIR-based segmentation workflow
- U-Net-style medical image segmentation
- Dice and IoU evaluation
- EEG signal preprocessing
- Seizure detection workflow
- Motor imagery BCI pipeline
- CSP and LDA classification
- Cross-validation and confusion matrix analysis
- FastAPI backend development
- Next.js and TypeScript frontend development
- Scientific visualization for model interpretation

---

## Disclaimer

This project is for research, education, and portfolio demonstration only.

It is not intended for clinical diagnosis, treatment decisions, or real-time patient monitoring.
