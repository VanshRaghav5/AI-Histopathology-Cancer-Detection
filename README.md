# 🧠 AI Histopathology Cancer Detection — Model Development & Verification

## 📌 Overview
This project focuses on the **deep learning model development, training, and verification** for **histopathology cancer detection**.  
The model classifies microscopic tissue slide images as **Benign** or **Malignant** and enhances interpretability using **Grad-CAM heatmaps**, highlighting regions contributing to predictions.

---

## 🗂️ Project Structure
AI-HISTOPATHOLOGY-CANCER-DETECTION/
│
├── data/                           # Dataset directory (raw and processed images)
│
├── gradcam_outputs/                # Generated Grad-CAM visualizations
│   ├── ductal_carcinoma_1_gradcam.png
│   ├── ductal_carcinoma_2_gradcam.png
│   ├── lobular_carcinoma_3452_gradcam.png
│   └── SOB_B_A-14-22549AB-40-001_gradcam.png
│
├── models/                         # Model weights and checkpoints
│   ├── checkpoint.pth
│   └── model_best.pth
│
├── scripts/                        # Helper scripts for inference and Grad-CAM generation
│   ├── generate_gradcam.py
│   └── gradcamoutput.png
│
├── src/                            # Source code
│   ├── model/
│   │   ├── __init__.py
│   │   ├── dataset.py              # Data loading and augmentation
│   │   ├── gradcam.py              # Grad-CAM implementation
│   │   ├── model.py                # CNN / ResNet model definition
│   │   ├── train.py                # Model training logic
│   │   ├── utils.py                # Utility functions (metrics, visualization, etc.)
│   │   └── __pycache__/            # Compiled cache
│   └── __pycache__/
│
├── text/                           # Notes, experiment logs, or documentation
│
├── notes/                          # Additional project notes
│
├── venv/                           # Virtual environment (optional)
│
├── .gitattributes
├── .gitignore
├── models.rar                      # Archived model files (for sharing/deployment)
└── README.md                       
---

## 🧩 Model Details
- **Architecture:** CNN-based / Transfer Learning (e.g., ResNet50)
- **Input Size:** 224×224 (RGB)
- **Output Classes:** `Benign`, `Malignant`
- **Framework:** PyTorch
- **Dataset:** Histopathology image dataset (multi-type cancer slides)
- **Final Model File:** `models/model_best.pth`

---

## ⚙️ Environment Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/AI-Histopathology-Cancer-Detection.git
   cd AI-Histopathology-Cancer-Detection
Create and activate virtual environment

bash
Copy code
python -m venv venv
source venv/bin/activate      # for Linux/Mac
venv\Scripts\activate         # for Windows
Install dependencies

bash
Copy code
pip install -r requirements.txt
🧪 Model Training (src/model/train.py)
The training pipeline includes:

Data preprocessing and augmentation

Train/validation data split using torch.utils.data.DataLoader

Model training loop with loss & accuracy tracking

Automatic checkpoint saving (model_best.pth)

▶️ Run Training
bash
Copy code
python src/model/train.py
After training completes, the best model weights are saved automatically in:

bash
Copy code
models/model_best.pth
🔍 Model Verification (scripts/generate_gradcam.py)
This script verifies model predictions and interpretability using Grad-CAM.

✳️ Features
Loads model_best.pth

Accepts a single histopathology image as input

Produces:

Original input image

Grad-CAM heatmap

Overlay visualization

Displays prediction with confidence score

▶️ Example Run
bash
Copy code
python scripts/generate_gradcam.py --image path_to_image.jpg
🧾 Example Output
makefile
Copy code
Prediction: Malignant
Confidence: 0.93
🖼️ Visualization Output
Generated Grad-CAM visualizations are saved under:
gradcam_outputs/

Example output files:
ductal_carcinoma_1_gradcam.png
ductal_carcinoma_2_gradcam.png
lobular_carcinoma_3452_gradcam.png
Each output includes:

Original Image

Grad-CAM Heatmap

Grad-CAM Overlay

Predicted Class + Confidence Score

⚙️ Tools & Libraries
Library	Purpose
PyTorch	Model training & inference
Torchvision	Pretrained ResNet architectures
OpenCV / PIL	Image processing
Matplotlib	Visualization
Grad-CAM	Model interpretability
NumPy / Pandas	Data handling

🧩 Results Summary
✅ Achieved high validation accuracy on test samples
✅ Grad-CAM correctly focused on cancerous tissue regions
✅ Model verified and ready for deployment as model_best.pth

📸 Example Grad-CAM Outputs
Input Image	Grad-CAM Overlay

💡 Usage Guide
🔹 To Train a New Model:
bash
Copy code
python src/model/train.py --epochs 50 --batch_size 32
🔹 To Verify Model Predictions:
bash
Copy code
python scripts/generate_gradcam.py --image path_to_image.jpg
🔹 To View Grad-CAM Results:
Open the saved image files inside:

Copy code
gradcam_outputs/
💡 Future Improvements
Expand to multi-class classification for different cancer subtypes

👤 Contributor
Vansh Raghav
Model Design • Training • Verification (Grad-CAM)

