<p align="center">
  <h1 align="center">😊 Facial Emotion Recognition with CNN</h1>
  <p align="center">
    A real-time facial emotion detection system using a custom Convolutional Neural Network trained on balanced, augmented image data — with live webcam inference.
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
    <img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
    <img src="https://img.shields.io/badge/Scikit--Learn-orange?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
    <img src="https://img.shields.io/badge/License-Academic-lightgrey" alt="Academic">
  </p>
</p>

---

## 📖 Overview

**Facial Emotion Recognition with CNN** is a machine learning project that detects human emotions from facial expressions in real time via webcam. It was developed as a term project for CENG463 — Introduction to Machine Learning.

The pipeline covers everything end-to-end:
1. **Preprocessing** — data balancing via downsampling and image augmentation
2. **Training** — a custom 4-block CNN with BatchNormalization, Dropout and EarlyStopping
3. **Inference** — real-time face detection + emotion classification using a webcam

### What we built

| Script | Description |
|--------|-------------|
| `codes/preprocessing.py` | Balances the dataset to 12,000 images per class using augmentation (flip, rotation, noise) and downsampling |
| `codes/model.py` | Builds and trains the CNN; saves the best model as `emotion_model.h5`; exports accuracy/loss graphs and a confusion matrix |
| `codes/realtime.py` | Loads the trained model and runs live emotion detection on webcam feed using Haar Cascade face detection |

### What we aimed for

- **Balanced training data**: Raw datasets are heavily imbalanced. Our preprocessing script ensures every class has exactly 12,000 samples.
- **Robust CNN architecture**: 4 convolutional blocks (32→64→128→256 filters) with BatchNorm and Dropout to fight overfitting.
- **Real-time usability**: The inference script runs smoothly on a webcam stream with confidence thresholding to avoid low-confidence predictions.
- **Reproducibility**: All paths and hyperparameters are defined as constants at the top of each script — easy to modify.

---

## 🏗️ Architecture

```
preprocessing.py          model.py                  realtime.py
──────────────────        ──────────────────────    ──────────────────────
 Data/                     Processed_Data/            emotion_model.h5
  ├── Angry/        ──►     ├── Angry/        ──►     + Haar Cascade
  ├── Fear/                 ├── Fear/                 + Webcam (cv2)
  ├── Happy/                ├── Happy/                │
  ├── Sad/                  ├── Sad/                  ▼
  └── Surprise/             └── Surprise/         Live Detection
                                 │                 (Bounding Box
                            CNN Training           + Emotion Label
                            (4 Conv Blocks)        + Confidence %)
                                 │
                            emotion_model.h5
                            Training_Results/
                              ├── accuracy_loss_graph.png
                              ├── confusion_matrix.png
                              └── classification_report.txt
```

---

## ✨ Features

- 🎯 **5-class emotion detection** — Angry, Fear, Happy, Sad, Surprise
- ⚖️ **Automatic dataset balancing** — augmentation for small classes, downsampling for large ones
- 🧠 **Custom CNN** — 4 conv blocks (32→64→128→256), BatchNorm, Dropout
- ⏹️ **EarlyStopping + ModelCheckpoint** — saves the best model, avoids wasted training time
- 📊 **Training visualizations** — accuracy/loss curves and confusion matrix exported automatically
- 🎥 **Real-time webcam inference** — Haar Cascade face detection + CNN classification, live
- 🔲 **Confidence threshold** — labels uncertain predictions as "Uncertain" below 40% confidence
- 🪞 **Mirror mode** — webcam feed is flipped so it feels natural

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Data | NumPy, Matplotlib, Seaborn |
| ML Utilities | Scikit-learn |
| Progress Bar | tqdm |
| Dataset | Human Face Emotions (Kaggle) |

---

## 👥 Team

| Name | Student ID |
|------|-----------|
| Mustafa Sefa Soysal | 23050111037 |
| Tevfik Han Parlak | 22050111025 |
| Jusif Cabbarzade | 21050141026 |

**Course:** CENG463 — Introduction to Machine Learning  
**University:** Ankara Yıldırım Beyazıt University (AYBU), 2025–2026 Fall Semester

---

## 📁 Project Structure

```
Facial-Emotion-Recognition-CNN/
│
├── codes/
│   ├── preprocessing.py        # Data balancing & augmentation
│   ├── model.py                # CNN definition, training & evaluation
│   └── realtime.py             # Webcam real-time inference
│
├── Data/                       # ⚠️ NOT included — download from Kaggle (see below)
│   ├── Angry/
│   ├── Fear/
│   ├── Happy/
│   ├── Sad/
│   └── Surprise/
│
├── Processed_Data/             # Generated by preprocessing.py (gitignored)
├── Training_Results/           # Generated by model.py (gitignored)
├── emotion_model.h5            # Generated by model.py (gitignored)
│
├── requirements.txt
├── data_set_url.txt            # Kaggle dataset link
└── README.md
```

---

## ⚙️ Prerequisites

- Python **3.8 or higher**
- A webcam (for `realtime.py`)
- GPU recommended for training (CPU works but is slow)

---

## 🚀 Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/sefasys/Facial-Emotion-Recognition-CNN.git
cd Facial-Emotion-Recognition-CNN
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the **Human Face Emotions** dataset from Kaggle:

👉 [https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)

Create a `Data/` folder in the project root and extract the dataset inside it.

### ⚠️ CRITICAL: Fix the "Suprise" Typo

The original dataset folder is named **`Suprise`** (missing an 'r'), but the code expects **`Surprise``. Rename it:

```bash
# Linux / macOS
mv Data/Suprise Data/Surprise

# Windows
ren Data\Suprise Surprise
```

### 5. Run preprocessing

Balances each class to 12,000 images and saves to `Processed_Data/`.

```bash
python codes/preprocessing.py
```

### 6. Train the model

Trains the CNN and saves the best model + evaluation results.

```bash
python codes/model.py
```

- **Duration:** ~20–40 minutes on GPU, longer on CPU
- **Outputs:** `emotion_model.h5`, `Training_Results/accuracy_loss_graph.png`, `Training_Results/confusion_matrix.png`, `Training_Results/classification_report.txt`

### 7. Run real-time detection

```bash
python codes/realtime.py
```

Press **`q`** to quit.

---

## 🤖 Model Architecture

```
Input: 48×48 Grayscale Image
│
├── Conv2D(32) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
├── Conv2D(64) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
├── Conv2D(128) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
├── Conv2D(256) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
│
├── Flatten
├── Dense(256) → BatchNorm → ReLU → Dropout(0.5)
│
└── Dense(5, softmax)   ← Output: [Angry, Fear, Happy, Sad, Surprise]
```

| Property | Value |
|----------|-------|
| Input Size | 48×48 Grayscale |
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Train/Val Split | 80% / 20% |
| Max Epochs | 50 (EarlyStopping patience=5) |
| Batch Size | 64 |

---

## 🔧 Troubleshooting

### `FileNotFoundError: .../Suprise`
Rename the dataset folder: `mv Data/Suprise Data/Surprise`

### `OSError: Data directory not found`
Make sure the `Data/` folder exists in the **same directory as the scripts**.

> **Note:** Scripts use relative paths. Run them from the project root, not from inside `codes/`.

### Webcam not opening
Change the camera index in `realtime.py`:
```python
cap = cv2.VideoCapture(0)  # Try 1 or 2 if 0 doesn't work
```

---

## 📄 License

This project is released for **academic and educational purposes**. No warranties expressed or implied.
