# 🧠 AI Histopathology Cancer Detection — Fusion Model (ResNet50 + VGG16)

## 📌 Overview
This repository contains the **Fusion Deep Learning Model** developed for **breast cancer histopathology image classification**.

The system identifies tissue as:

- **Benign (Non-Cancerous)**
- **Malignant (Cancerous)**

To achieve high accuracy and better tissue-level understanding, the model uses:

### 🔥 Fusion Architecture
- **ResNet50** (deep semantic features)  
- **VGG16** (fine-grained texture features)  
- Feature vectors are **concatenated** and passed into custom fully connected layers.

### 🎯 Explainability Included
To ensure transparency, the project includes:
- **Grad-CAM**
- **Guided Grad-CAM**
- **Joint Grad-CAM** (Fusion model explainability combining both backbones)

---

## 🗂 Project Structure

```bash
AI-HISTOPATHOLOGY-CANCER-DETECTION/
│
├── data/                         
│
├── gradcam_outputs/              
│   ├── *_gradcam.png
│   ├── *_guided_gradcam.png
│   └── *_joint_gradcam.png
│
├── models/
│   ├── checkpoint.pth
│   └── model_best.pth     # Download externally (677 MB)
│
├── scripts/
│   ├── generate_gradcam.py
│   ├── generate_guided_gradcam.py
│   └── generate_joint_gradcam.py
│
├── src/
│   ├── model/
│   │   ├── dataset.py
│   │   ├── model.py                # Fusion Model
│   │   ├── train.py
│   │   ├── utils.py
│   │   └── gradcam.py
│
├── notes/
├── text/
├── venv/
└── README.md
```

---

## 📥 Download Model (677 MB)

GitHub cannot host files >100 MB, so the trained fusion model is hosted externally.

👉 **Download `model_best.pth`:**  
https://drive.google.com/file/d/1-VyqwdJ9250jYR0rp7tKVGReao-9Q1fD/view?usp=drive_link

---

## 🧩 Model Details

| Feature | Description |
|--------|-------------|
| **Architecture** | ResNet50 + VGG16 Fusion |
| **Fusion Method** | Feature Concatenation → Dense Layers |
| **Input Size** | 224×224 RGB |
| **Output Classes** | Benign (0), Malignant (1) |
| **Loss Function** | CrossEntropy |
| **Best Epoch** | 4 |
| **Explainability Tools** | Grad-CAM, Guided Grad-CAM, Joint Grad-CAM |

---

## 📊 Best Validation Metrics (Epoch 4)

```
Val Accuracy: 94.17%
Train Accuracy: 90.45%
F1 Score: ~0.94
AUC-ROC: ~0.97+
```

---

## ⚙️ Environment Setup

### Clone the Repository
```bash
git clone https://github.com/<your-username>/AI-Histopathology-Cancer-Detection.git
cd AI-Histopathology-Cancer-Detection
```

### Create Virtual Environment
```bash
python -m venv venv
venv\Scriptsctivate      # Windows
source venv/bin/activate   # Linux/Mac
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Training the Fusion Model

```bash
python -m src.model.train --arch fusion --epochs 15 --batch_size 8 --lr 0.001 --pretrained
```

Best model saved at:
```
models/model_best.pth
```

---

## 🔍 Explainability Tools

### Standard Grad-CAM
```bash
python scripts/generate_gradcam.py
```

### Guided Grad-CAM
```bash
python scripts/generate_guided_gradcam.py
```

### Joint Grad-CAM (Fusion)
```bash
python scripts/generate_joint_gradcam.py
```

Output saved in:
```
gradcam_outputs/
```

---

## 📚 Technologies Used
- PyTorch  
- Torchvision  
- OpenCV  
- Matplotlib  
- Tkinter  
- NumPy  
- Grad-CAM

---

## 👤 Contributor
**Vansh Raghav**
_Model Fusion • Training • Verification • Explainability (Grad-CAM)_

