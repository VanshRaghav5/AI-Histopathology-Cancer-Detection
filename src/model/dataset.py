import os
from typing import Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class HistopathDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        binary_mapping: Dict[str, int] = {
           
            'BREAST_ADENOSIS': 0,
            'BREAST_FIBRODENOMA': 0,
            'BREAST_PYLLODES_TUMOR': 0, 
            'BREAST_TUBULAR_ADENOMA': 0,
       
            'BREAST_DUCTAL_CARCINOMA': 1,
            'BREAST_LOBULAR_CARCINOMA': 1,
            'BREAST_MUCINOUS_CARCINOMA': 1,
            'BREAST_PAPILLARY_CARCINOMA': 1,
           
        }

        for folder_name, label in binary_mapping.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"Warning: Folder '{folder_path}' not found. Skipping.")
                continue
            folder_samples = 0
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(folder_path, fname), label))
                    folder_samples += 1
            print(f"Loaded {folder_samples} samples from '{folder_name}' (label {label})")

        if not self.samples:
            raise ValueError(f"No valid images found in {root_dir}. Check folder structure and image files.")

        total_benign = sum(1 for _, lbl in self.samples if lbl == 0)
        total_malignant = sum(1 for _, lbl in self.samples if lbl == 1)
        print(f"Total dataset: {len(self.samples)} samples ({total_benign} benign, {total_malignant} malignant)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(train: bool = True, size: int = 224):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize
        ])


def load_dataset(root_dir: str, train: bool = True, size: int = 224) -> HistopathDataset:
    return HistopathDataset(root_dir, transform=get_transforms(train=train, size=size))


def create_train_val_loaders(root_dir: str,
                             batch_size: int = 32,
                             val_ratio: float = 0.2,
                             size: int = 224,
                             num_workers: int = 4,
                             seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    torch.manual_seed(seed)
    full_dataset = load_dataset(root_dir, train=True, size=size)  
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(seed))

    val_dataset.dataset.transform = get_transforms(train=False, size=size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Created loaders: Train={len(train_loader.dataset)} samples, Val={len(val_loader.dataset)} samples")
    return train_loader, val_loader