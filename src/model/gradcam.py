import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        for module in self.model.modules():
            if isinstance(module, self.target_layer):
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))

    def generate(self, input_image, class_idx=None):
        self.model.eval()
        logit = self.model(input_image)
        if class_idx is None:
            class_idx = torch.argmax(logit, dim=1).item()
        score = logit[:, class_idx].squeeze()
        self.model.zero_grad()
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0][i]

        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        return cam.cpu().detach().numpy(), class_idx

    def __del__(self):
        for hook in self.hooks:
            hook.remove()

def get_last_conv_layer(model):

    return model.features[-1] 

def overlay_heatmap(original_img, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap * alpha + original_img
    cam_img = cam_img / np.max(cam_img)
    return np.uint8(255 * cam_img)

def generate_gradcam(model_path, image_path, output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)['model_state_dict'] 
    model = get_model(num_classes=2, pretrained=False)  
    model.load_state_dict(model)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    target_layer = get_last_conv_layer(model)
    grad_cam = GradCAM(model, type(target_layer)) 

    cam, pred_class = grad_cam.generate(input_tensor)
    cam = cv2.resize(cam, (224, 224))

    original = np.array(image.resize((224, 224)))
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    heatmap_overlay = overlay_heatmap(original, cam)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Original (Pred: {'Cancerous' if pred_class == 1 else 'Non-Cancerous'})")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(heatmap_overlay)
    plt.title("Overlay")
    plt.axis('off')

    if output_path:
        plt.savefig(output_path)
    plt.show()