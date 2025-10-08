# src/model/train.py
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.model.dataset import create_train_val_loaders
from src.model.model import get_model
from src.model.utils import save_checkpoint, load_checkpoint  # Assuming utils.py has these


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(outputs, labels, num_classes=2):
    preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    acc = accuracy_score(labels, np.argmax(preds, axis=1))
    f1 = f1_score(labels, np.argmax(preds, axis=1), average='binary') 
    auc = roc_auc_score(labels, preds[:, 1]) 
    return acc, f1, auc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_loader, val_loader = create_train_val_loaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=0.2,
        size=224,
        num_workers=args.num_workers,
        seed=args.seed
    )

    model = get_model(name=args.arch, num_classes=2, pretrained=args.pretrained)
    model = model.to(device)

    total_samples = 6442  
    num_benign = 1013 
    num_malignant = 5429  
    weight_benign = num_malignant / total_samples 
    weights = ```````````torch.tensor([total_samples / (2 * num_benign), total_samples / (2 * num_malignant)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = load_checkpoint(args.checkpoint_path)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']


    for epoch in range(start_epoch, args.epochs):
   
        model.train()
        train_loss = 0.0
        train_outputs, train_labels = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_outputs.append(outputs)
            train_labels.append(labels)
            pbar.set_postfix({'loss': loss.item()})
        train_acc, train_f1, train_auc = compute_metrics(
            torch.cat(train_outputs), torch.cat(train_labels)
        )

       
        model.eval()
        val_loss = 0.0
        val_outputs, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_outputs.append(outputs)
                val_labels.append(labels)
        val_acc, val_f1, val_auc = compute_metrics(
            torch.cat(val_outputs), torch.cat(val_labels)
        )

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        is_best = val_acc > best_acc
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': max(best_acc, val_acc),
            'train_acc': train_acc,
            'val_acc': val_acc
        }, is_best, args.save_dir)
        best_acc = max(best_acc, val_acc)

    print("Training completed. Best model saved in models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Histopathology Cancer Detector")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--arch', type=str, default='mobilenet_v2', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='models/checkpoint.pth', help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default='models', help='Path to save models')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    train(args)