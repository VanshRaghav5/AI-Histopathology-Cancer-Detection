import torch
import torch.nn.functional as F
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# ----------------------------------------------------
# Make sure project root is on sys.path
# ----------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.model import get_model
from src.model.dataset import get_transforms


# ----------------------------------------------------
# Grad-CAM using ResNet branch of Fusion model
# ----------------------------------------------------
def get_gradcam_standard(model, input_tensor, target_class=None):
    """
    Computes Grad-CAM using the last conv block of ResNet50
    inside the fusion model (model.resnet.layer4[-1]).
    """
    model.eval()
    input_tensor.requires_grad_(True)

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        # keep activations with grad support
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # ✅ For fusion model: use ResNet's last conv block
    target_layer = model.resnet.layer4[-1]

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    if activations is None or gradients is None:
        raise ValueError("Could not capture activations or gradients.")

    # GAP over spatial dims -> channel-wise weights
    # gradients: [B, C, H, W]
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam = torch.sum(weights * activations, dim=1).squeeze()     # [H, W]
    cam = F.relu(cam)

    # Resize CAM to 224x224
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze().detach().cpu().numpy()

    # Normalize between 0 and 1
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def visualize_gradcam(image_path, model, transform, device, temperature=2.0):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # -----------------------------
    # Forward pass & confidence
    # -----------------------------
    with torch.no_grad():
        output = model(input_tensor)  # [1, 2]

        # ✅ Numerically stable + temp scaling
        # PyTorch softmax is already stable, no need for extra shifting
        probs_tensor = F.softmax(output / temperature, dim=1)[0]
        probs = probs_tensor.cpu().numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    pred_label = '🧬 Malignant (Cancerous)' if pred_class == 1 else '🌿 Benign (Non-Cancerous)'

    print(f"\n🖼️ Image: {os.path.basename(image_path)}")
    print(f"Prediction: {pred_label} (Class {pred_class})")
    print(f"Confidence: {confidence:.4f}")
    print(f"Benign Prob: {probs[0]:.4f} | Malignant Prob: {probs[1]:.4f}")

    # -----------------------------
    # Grad-CAM
    # -----------------------------
    try:
        gradcam = get_gradcam_standard(
            model, input_tensor.clone().requires_grad_(True), pred_class
        )
        print("✅ Grad-CAM computed successfully!")
    except Exception as e:
        print(f"⚠️ Error computing Grad-CAM: {e}")
        gradcam = np.ones((224, 224)) * 0.5

    img_np = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.4 * heatmap + 0.6 * img_np
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    im1 = axes[1].imshow(gradcam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay (Pred: {pred_label}, Conf: {confidence:.2f})')
    axes[2].axis('off')

    plt.tight_layout()

    output_dir = os.path.join(project_root, "gradcam_outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.splitext(os.path.basename(image_path))[0] + "_gradcam.png"
    output_path = os.path.join(output_dir, output_name)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"💾 Saved Grad-CAM visualization as: {output_path}")


def main():
    print("🚀 Grad-CAM Visualizer — Interactive Mode")

    model_path = os.path.join(project_root, 'models', 'model_best.pth')
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ✅ Explicitly load fusion model
    model = get_model(name="fusion", num_classes=2, pretrained=False)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    transform = get_transforms(train=False, size=224)

    root = tk.Tk()
    root.withdraw()

    print("\n🧩 Upload histopathology images one by one to visualize Grad-CAM.")
    print("Press Cancel to exit.\n")

    while True:
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")]
        )

        if not file_path:
            print("\n✅ Session ended. Exiting Grad-CAM viewer.")
            break

        visualize_gradcam(file_path, model, transform, device, temperature=2.0)


if __name__ == "__main__":
    main()
