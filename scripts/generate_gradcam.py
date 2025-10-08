import argparse
import torch
import torch.nn.functional as F
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.model import get_model
from src.model.dataset import get_transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_gradcam_standard(model, input_tensor, target_class=None):

    model.eval()

    input_tensor.requires_grad_(True)

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()  

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]  

    target_layer = model.features[-4]  
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
     
        output = model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        fwd_handle.remove()
        bwd_handle.remove()

    except Exception as e:
        fwd_handle.remove()
        bwd_handle.remove()
        raise e

    if activations is None or gradients is None:
        raise ValueError(f"Failed to capture activations/gradients from {target_layer}. Try different layer.")

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze()  
    cam = F.relu(cam) 

    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),  
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam) 

    return cam


def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualization for histopathology image.')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model_path', type=str, required=True,
                        default='models/model_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output_path', type=str, default='gradcam_output.png',
                        help='Output visualization path')
    args = parser.parse_args()

    print(f"Debug: Parsed args - Image: {args.image_path}, Model: {args.model_path}, Output: {args.output_path}")

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model(num_classes=2, pretrained=False)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    transform = get_transforms(train=False, size=224)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0].cpu().numpy()
        pred_class = output.argmax(dim=1).item()
        confidence = probs[pred_class]

    pred_label = 'Malignant (Cancerous)' if pred_class == 1 else 'Benign (Non-Cancerous)'
    print(f"Generated Grad-CAM for: {args.image_path}")
    print(f"Prediction: {pred_label} (Class {pred_class}) with confidence {confidence:.4f}")
    print(f"Benign Prob: {probs[0]:.4f} | Malignant Prob: {probs[1]:.4f}")

    gradcam = None
    try:
        print("Computing Grad-CAM...")
        gradcam = get_gradcam_standard(model, input_tensor.clone().requires_grad_(True), pred_class)
        print("Grad-CAM computed successfully!")
    except Exception as e:
        print(f"Error computing Grad-CAM: {e}")
        print("Using fallback uniform heatmap for visualization.")
        gradcam = np.ones((224, 224)) * 0.5  
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())

    img_np = np.array(image.resize((224, 224)))

    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = 0.4 * heatmap.astype(np.float32) + 0.6 * img_np.astype(np.float32)
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

    if not args.output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        args.output_path += '.png'
    plt.savefig(args.output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {args.output_path}")
    print("Success! Open the PNG to see the model's focus areas (red/yellow = high attention).")


if __name__ == "__main__":
    main()