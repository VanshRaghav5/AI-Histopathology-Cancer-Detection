import torch
import torch.nn.functional as F
from src.model.model import get_model
from src.model.dataset import get_transforms
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = get_model(name="fusion", num_classes=2, pretrained=False)

checkpoint_path = 'models/model_best.pth'

if not os.path.exists(checkpoint_path):
    print("❌ Model checkpoint not found in models/model_best.pth")
    exit()

checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

print(f"✅ Loaded model from {checkpoint_path}")


best_acc = checkpoint.get('best_acc', 'N/A')
best_f1 = checkpoint.get('best_f1', 'N/A')
best_auc = checkpoint.get('best_auc', 'N/A')
epoch = checkpoint.get('epoch', 'N/A')

print(f"📊 Best Validation — Acc: {best_acc}, F1: {best_f1}, AUC: {best_auc}, Epoch: {epoch}")


transform = get_transforms(train=False, size=224)


root = tk.Tk()
root.withdraw()  

print("\n🧩 Model ready! Upload images one by one to test predictions.")
print("Press 'Cancel' in the file dialog when done.\n")

while True:
    file_path = filedialog.askopenfilename(
        title="Select an image to test",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")]
    )

    if not file_path:
        print("✅ Testing finished.")
        break

  
    try:
        image = Image.open(file_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Error opening image: {e}")
        continue

    input_tensor = transform(image).unsqueeze(0).to(device)

   
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0].cpu().numpy()
        class_idx = torch.argmax(output, dim=1).item()
        confidence = probs[class_idx]

    pred_label = '🧬 Malignant (Cancerous)' if class_idx == 1 else '🌿 Benign (Non-Cancerous)'

    print(f"\n🖼️ Image: {os.path.basename(file_path)}")
    print(f"Prediction: {pred_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Benign Prob: {probs[0]:.4f} | Malignant Prob: {probs[1]:.4f}")
    print("-" * 50)
