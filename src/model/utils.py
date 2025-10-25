import torch
import os
from typing import Dict, Any, Optional

def save_checkpoint(state: Dict[str, Any], is_best: bool, save_dir: str):
    """
    Save model checkpoint
    
    Args:
        state: Model state dictionary
        is_best: Whether this is the best model so far
        save_dir: Directory to save checkpoints
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(state, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_path)
        print(f"New best model saved to {best_path}")

def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Checkpoint dictionary or None if loading failed
    """
    try:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
            return None
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
