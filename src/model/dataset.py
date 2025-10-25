import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List

class HistopathologyDataset(Dataset):
    """Dataset for histopathology images"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_train_val_loaders(
    root_dir: str,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    size: int = 224,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        root_dir: Root directory containing data
        batch_size: Batch size for data loaders
        val_ratio: Ratio of data to use for validation
        size: Image size for resizing
        num_workers: Number of worker processes
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Collect image paths and labels
    image_paths = []
    labels = []
    
    # Assuming directory structure: root_dir/benign/ and root_dir/malignant/
    benign_dir = os.path.join(root_dir, 'benign')
    malignant_dir = os.path.join(root_dir, 'malignant')
    
    if os.path.exists(benign_dir):
        for filename in os.listdir(benign_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(benign_dir, filename))
                labels.append(0)  # 0 for benign
    
    if os.path.exists(malignant_dir):
        for filename in os.listdir(malignant_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(malignant_dir, filename))
                labels.append(1)  # 1 for malignant
    
    if not image_paths:
        raise ValueError(f"No images found in {root_dir}")
    
    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=val_ratio, random_state=seed, stratify=labels
    )
    
    # Create datasets
    train_dataset = HistopathologyDataset(train_paths, train_labels, train_transform)
    val_dataset = HistopathologyDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader
