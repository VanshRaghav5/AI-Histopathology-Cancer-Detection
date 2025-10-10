# AI-Histopathology-Cancer-Detection

Binary classifier for breast cancer detection in histopathology images using MobileNetV2 (PyTorch).
Dataset: BreakHis,Lung and Colon (benign/malignant).

## Setup
- Install: `pip install -r requirements.txt`
- Train: `python scripts/train.py`
- Evaluate: `python scripts/evaluate.py`
- Grad-CAM: `python scripts/generate_gradcam.py --image_path <path> ...`

## Results
- Accuracy: 98%+ on test set.
- Explainable: Grad-CAM heatmaps for model focus.

## Structure
- src/: Model and dataset code.
- scripts/: Training, eval, visualization.
