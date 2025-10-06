import os
import torch

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        torch.save(state, os.path.join(save_dir, 'model_best.pth'))

def load_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        return torch.load(checkpoint_path, map_location='cpu')
    return None