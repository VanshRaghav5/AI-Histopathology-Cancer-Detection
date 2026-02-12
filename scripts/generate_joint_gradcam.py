import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.model import get_model
from src.model.dataset import get_transforms



def compute_gradcam(model, input_tensor, target_class, target_layer):
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    fwd = target_layer.register_forward_hook(forward_hook)
    bwd = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()

    class_score = output[0, target_class]
    class_score.backward()

    fwd.remove()
    bwd.remove()

    if activations is None or gradients is None:
        raise RuntimeError("❌ Gradients or activations missing for target layer.")

    weights = gradients.mean(dim=(2, 3), keepdim=True)      
    cam = (weights * activations).sum(dim=1).squeeze()      
    cam = F.relu(cam)

    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze().detach().cpu().numpy()

    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam



def generate_joint_gradcam(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

   
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()

    pred_class = int(np.argmax(probs))
    confidence = probs[pred_class]

    pred_label = "🧬 Malignant" if pred_class == 1 else "🌿 Benign"
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Prediction: {pred_label}, Confidence: {confidence:.4f}")

    print("🔍 Computing ResNet Grad-CAM...")
    resnet_layer = model.resnet.layer4[-1]  
    resnet_cam = compute_gradcam(model, input_tensor.clone().requires_grad_(True),
                                 pred_class, resnet_layer)

    print("🔍 Computing VGG16 Grad-CAM...")
  
    vgg_layer = model.vgg.features[-1]
    vgg_cam = compute_gradcam(model, input_tensor.clone().requires_grad_(True),
                              pred_class, vgg_layer)

    joint_cam = (resnet_cam + vgg_cam) / 2.0
    joint_cam = (joint_cam - joint_cam.min()) / (joint_cam.max() + 1e-8)

    orig = np.array(image.resize((224, 224)))

    heat_resnet = cv2.applyColorMap(np.uint8(resnet_cam * 255), cv2.COLORMAP_JET)
    heat_vgg = cv2.applyColorMap(np.uint8(vgg_cam * 255), cv2.COLORMAP_JET)
    heat_joint = cv2.applyColorMap(np.uint8(joint_cam * 255), cv2.COLORMAP_JET)

    heat_resnet = cv2.cvtColor(heat_resnet, cv2.COLOR_BGR2RGB)
    heat_vgg = cv2.cvtColor(heat_vgg, cv2.COLOR_BGR2RGB)
    heat_joint = cv2.cvtColor(heat_joint, cv2.COLOR_BGR2RGB)

    overlay_joint = (0.4 * heat_joint + 0.6 * orig).astype(np.uint8)

    
    fig, axes = plt.subplots(1, 4, figsize=(25, 6))

    axes[0].imshow(orig)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heat_resnet)
    axes[1].set_title("ResNet Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(heat_vgg)
    axes[2].set_title("VGG16 Grad-CAM")
    axes[2].axis("off")

    axes[3].imshow(overlay_joint)
    axes[3].set_title(f"Joint Grad-CAM\n{pred_label} ({confidence:.2f})")
    axes[3].axis("off")

    plt.tight_layout()


    output_dir = os.path.join(project_root, "gradcam_outputs")
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_joint_gradcam.png"
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"💾 Saved Joint Grad-CAM to: {save_path}")


def main():
    print("🚀 Joint Grad-CAM Visualizer — Fusion Model (ResNet + VGG)")

    model_path = os.path.join(project_root, "models", "model_best.pth")
    if not os.path.exists(model_path):
        print("❌ model_best.pth NOT FOUND.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model = get_model(name="fusion", num_classes=2, pretrained=False)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    transform = get_transforms(train=False, size=224)

    root = tk.Tk()
    root.withdraw()

    while True:
        f = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.tif;*.bmp")]
        )
        if not f:
            print("Exiting.")
            break

        generate_joint_gradcam(f, model, transform, device)


if __name__ == "__main__":
    main()
