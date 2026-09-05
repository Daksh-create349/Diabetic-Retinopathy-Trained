# Diabetic Retinopathy Screening using Deep Learning

An AI-based diabetic retinopathy (DR) screening system that analyzes retinal fundus images and classifies the severity of diabetic retinopathy into five grades.

The project is designed as a screening-support system with a focus on detecting **referable diabetic retinopathy**, while also providing explainable and measurable AI predictions.

---

## Overview

Diabetic Retinopathy is a diabetes-related eye disease that can cause vision loss if it is not detected and treated at an early stage.

This project uses deep learning to analyze retinal fundus images and classify them into five diabetic retinopathy severity levels:

| Grade | Description |
|------:|-------------|
| 0 | No Diabetic Retinopathy |
| 1 | Mild Diabetic Retinopathy |
| 2 | Moderate Diabetic Retinopathy |
| 3 | Severe Diabetic Retinopathy |
| 4 | Proliferative Diabetic Retinopathy |

The system also converts the five-class prediction into a screening decision:

- **Grade 0–1 → Non-Referable**
- **Grade 2–4 → Referable**

---

## Dataset

The project uses an APTOS 2019 diabetic retinopathy dataset containing labeled retinal fundus images.

### Dataset Split

The dataset used in the project contains:

- **Training images:** 2930
- **Validation images:** 366
- **Test images:** 366

The dataset contains five diabetic retinopathy severity classes ranging from Grade 0 to Grade 4.

---

## Project Pipeline

```text
Retinal Fundus Image
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
EfficientNetB3
        ↓
5-Class DR Classification
        ↓
Grade 0 / 1 / 2 / 3 / 4
        ↓
Referable DR Decision
        ↓
Sensitivity / Specificity Evaluation
```

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab
- NVIDIA T4 GPU
- Kaggle Dataset

---

## Model

### EfficientNetB3

The project uses EfficientNetB3 with ImageNet pretrained weights.

Transfer learning is used to take advantage of pretrained visual features and adapt the network to diabetic retinopathy classification.

### Model Configuration

- **Architecture:** EfficientNetB3
- **Input resolution:** 384 × 384
- **Output classes:** 5
- **Optimizer:** Adam
- **Data augmentation:** Horizontal Flip, Rotation, Zoom and Contrast
- **Class balancing:** Balanced class weights
- **Mixed precision:** Enabled for NVIDIA T4 GPU
- **Focal Loss:** Used in the improved V2 model

---

## Image Preprocessing

The preprocessing pipeline includes:

- Reading retinal fundus images using OpenCV
- Removing unnecessary black borders around the retinal field
- Converting images from BGR to RGB
- Resizing images to 384 × 384
- Preparing images for EfficientNet processing

The improved pipeline uses TensorFlow's `tf.data` pipeline so that images are processed batch-by-batch instead of loading the complete dataset into RAM.

---

## Model Versions

### V1 — Baseline Model

The first model was trained using EfficientNetB3 with transfer learning, augmentation and class weighting.

**Best validation accuracy:** 73.22%

The V1 model also achieved strong referable DR screening performance after threshold optimization.

### V2 — Improved Model

The second model introduced Focal Loss to better handle difficult and imbalanced diabetic retinopathy classes.

**Best validation accuracy:** 80.87%

The V2 model significantly improved over the V1 baseline.

---

## Validation Performance

### V1 — Five-Class Classification

**Best validation accuracy:** 73.22%

The main difficulty was distinguishing between the more severe DR classes, particularly:

- Moderate DR
- Severe DR
- Proliferative DR

### V2 — Five-Class Classification

**Best validation accuracy:** 80.87%

V2 improved classification performance, particularly for the Moderate DR class.

---

## Referable DR Screening

For screening purposes, the five DR grades are grouped as:

```text
Grade 0 + Grade 1
        ↓
Non-Referable

Grade 2 + Grade 3 + Grade 4
        ↓
Referable
```

A threshold optimization was performed using the validation set to improve screening sensitivity while maintaining good specificity.

### Locked Screening Threshold

**Referable probability threshold = 0.33**

### Final Held-Out Test Performance

Using the threshold selected on the validation set, the model achieved on the untouched test set:

- **Sensitivity:** 93.43%
- **Specificity:** 93.01%

### Test Confusion Matrix

|                    | Predicted: Non-Referable | Predicted: Referable |
|--------------------|:-------------------------:|:----------------------:|
| **Actual: Non-Referable** | 213 | 16 |
| **Actual: Referable**     | 9   | 128 |

This corresponds to:

- **True Negatives:** 213
- **False Positives:** 16
- **False Negatives:** 9
- **True Positives:** 128

---

## Why Sensitivity and Specificity Matter

For a screening system, overall five-class accuracy alone is not sufficient.

The system should avoid missing patients who may have referable diabetic retinopathy.

Therefore, important screening metrics include:

**Sensitivity**
The percentage of truly referable cases correctly identified by the system.

**Specificity**
The percentage of non-referable cases correctly identified as non-referable.

The current screening configuration achieved:

- Sensitivity > 90%
- Specificity > 85%

on the held-out test set.

---

## Training Strategy

The project uses a staged training strategy.

### Stage 1

The pretrained EfficientNetB3 backbone is initially frozen while the classification head learns the diabetic retinopathy task.

### Stage 2

A deeper fine-tuning strategy is explored using a smaller learning rate.

### V2 Improvement

Focal Loss is introduced to give more attention to difficult examples and underrepresented disease classes.

---

## Training Environment

Recommended environment:

- Google Colab
- NVIDIA T4 GPU
- Python 3
- TensorFlow

Mixed precision is enabled to improve GPU efficiency on compatible hardware.

---

## Repository Structure

```text
.
├── README.md
├── notebooks/
│   └── diabetic_retinopathy_training.ipynb
│
├── models/
│   ├── MODEL_V1_73pct_backup.keras
│   ├── MODEL_V2_80pct_backup.keras
│   └── MODEL_V2_high_accuracy.keras
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_accuracy.png
│   └── training_loss.png
│
└── requirements.txt
```

Model files may be stored separately from the Git repository because of their large file size.

---

## How to Run

### 1. Open the notebook

Open the training notebook in Google Colab.

### 2. Enable GPU

In Google Colab:

```
Runtime → Change runtime type → T4 GPU
```

### 3. Configure Kaggle

Generate a Kaggle API token and provide it to the notebook when prompted.

Do not commit Kaggle credentials or tokens to GitHub.

### 4. Download the dataset

The notebook downloads and extracts the APTOS dataset.

### 5. Train the model

Run the notebook cells sequentially.

The training pipeline automatically:

- Loads retinal images
- Performs preprocessing
- Applies augmentation
- Trains EfficientNetB3
- Saves the best model
- Tracks validation performance
- Performs fine-tuning
- Evaluates classification performance

---

## Evaluation Metrics

The project evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Sensitivity
- Specificity

The final screening evaluation is performed on an untouched test set.

---

## Model Checkpointing

Best models are automatically saved during training.

Example:

```text
models/
├── MODEL_V1_73pct_backup.keras
├── MODEL_V2_80pct_backup.keras
└── MODEL_V2_high_accuracy.keras
```

This allows training to be resumed from a saved checkpoint if a Colab runtime disconnects or crashes.

---

## Current Results

| Model | Best Validation Accuracy |
|-------|---------------------------|
| V1 Baseline | 73.22% |
| V2 Focal Loss | 80.87% |

### Referable DR Screening

| Metric | Test Result |
|--------|-------------|
| Sensitivity | 93.43% |
| Specificity | 93.01% |

---

## Future Work

The next stages of the project include:

- Further improving five-class DR classification
- Better separation of Severe and Proliferative DR
- Image quality assessment
- Retinal image enhancement
- Lesion detection
- Blood vessel segmentation
- Optic disc and fovea localization
- Grad-CAM explainability
- Confidence calibration
- Automated screening reports
- External validation using additional datasets
- MATLAB/Simulink-based large-scale healthcare deployment simulation

---

## Clinical Disclaimer

This project is intended as an AI-assisted diabetic retinopathy screening and research prototype.

It is not intended to replace a qualified ophthalmologist or serve as a standalone clinical diagnosis system.

All predictions should be reviewed by an appropriately qualified healthcare professional before clinical decisions are made.

---

## License

This project is intended for educational, research and hackathon purposes.

Please ensure that the licenses and usage terms of the underlying datasets are followed when using or redistributing the data.

---

## Author

**Daksh Srivastava**
B.Tech Computer Science & Engineering

---

## Acknowledgements

- APTOS 2019 Diabetic Retinopathy Dataset
- TensorFlow / Keras
- OpenCV
- Scikit-learn
- Google Colab
- Kaggle
