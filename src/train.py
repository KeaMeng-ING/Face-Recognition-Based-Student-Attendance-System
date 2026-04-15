"""
SIMPLIFIED Training Script with Debugging
Lower learning rate and better stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

from mobilefacenet import MobileFaceNet
from dataset import LFWSubset, transform
from main import image_paths, labels, label_to_idx


class ArcFaceLoss(nn.Module):
    """Simplified ArcFace with extra stability"""
    def __init__(self, embedding_size=512, num_classes=158, s=30.0, m=0.30):
        super(ArcFaceLoss, self).__init__()
        self.s = s  # Reduced from 64 to 30 for stability
        self.m = m  # Reduced from 0.5 to 0.3 for easier training
        self.num_classes = num_classes
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, embeddings, labels):
        # Normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        weight = nn.functional.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = nn.functional.linear(embeddings, weight)
        
        # Compute sine
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, 1e-7, 1.0))
        
        # Add margin: cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot for target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        return nn.functional.cross_entropy(output, labels)


def train_one_epoch(model, criterion, dataloader, optimizer, device, epoch):
    model.train()
    criterion.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, labels_batch) in enumerate(pbar):
        images = images.to(device)
        labels_batch = labels_batch.to(device)
        
        # Forward
        embeddings = model(images)
        loss = criterion(embeddings, labels_batch)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\n❌ NaN loss detected at batch {batch_idx}!")
            print(f"   Embedding stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, mean={embeddings.mean():.4f}")
            return None, None
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # Stats
        running_loss += loss.item()
        
        # Accuracy
        with torch.no_grad():
            emb_norm = nn.functional.normalize(embeddings, p=2, dim=1)
            w_norm = nn.functional.normalize(criterion.weight, p=2, dim=1)
            cosine = nn.functional.linear(emb_norm, w_norm)
            _, predicted = torch.max(cosine, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100 * correct / total:.1f}%'
        })
    
    return running_loss / len(dataloader), 100 * correct / total


def validate(model, criterion, dataloader, device):
    model.eval()
    criterion.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels_batch in dataloader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            
            embeddings = model(images)
            loss = criterion(embeddings, labels_batch)
            
            running_loss += loss.item()
            
            emb_norm = nn.functional.normalize(embeddings, p=2, dim=1)
            w_norm = nn.functional.normalize(criterion.weight, p=2, dim=1)
            cosine = nn.functional.linear(emb_norm, w_norm)
            _, predicted = torch.max(cosine, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
    
    return running_loss / len(dataloader), 100 * correct / total


def main():
    # SIMPLIFIED CONFIG
    config = {
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.01,  # REDUCED from 0.1
        'embedding_size': 512,
        'num_workers': 2,  # Reduced to avoid warnings
        'arcface_s': 30.0,  # Reduced from 64.0
        'arcface_m': 0.30,  # Reduced from 0.50
    }
    
    print("=" * 60)
    print("SIMPLIFIED MobileFaceNet Training")
    print("Lower LR + Easier ArcFace settings for stability")
    print("=" * 60)
    
    # Verify data first
    print("\nData verification:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Total labels: {len(labels)}")
    print(f"  Unique classes: {len(label_to_idx)}")
    print(f"  Label range: [{min(labels)}, {max(labels)}]")
    
    if min(labels) != 0 or max(labels) != len(label_to_idx) - 1:
        print("\n❌ ERROR: Labels not in correct range!")
        print("   Run diagnose_data.py first to debug")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Config: {config}")
    
    # Dataset
    dataset = LFWSubset(image_paths, labels, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model
    model = MobileFaceNet(embedding_size=config['embedding_size'])
    model = model.to(device)
    
    criterion = ArcFaceLoss(
        embedding_size=config['embedding_size'],
        num_classes=len(label_to_idx),
        s=config['arcface_s'],
        m=config['arcface_m']
    )
    criterion = criterion.to(device)
    
    print(f"\nModel params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"ArcFace: s={config['arcface_s']}, m={config['arcface_m']}")
    
    # Optimizer with warmup
    optimizer = optim.SGD(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config['learning_rate'],
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[15, 25],
        gamma=0.1
    )
    
    # Training
    os.makedirs('./checkpoints', exist_ok=True)
    
    best_val_acc = 0.0
    patience = 0
    max_patience = 10
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch
        )
        
        if train_loss is None:
            print("\n❌ Training failed due to NaN. Stopping.")
            break
        
        # Validate
        val_loss, val_acc = validate(model, criterion, val_loader, device)
        
        # LR
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  LR: {lr:.6f}")
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'val_acc': val_acc,
            }, './checkpoints/best_model.pth')
            print(f"  ✓ New best: {val_acc:.2f}%")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\n⚠️  Early stopping (no improvement for {max_patience} epochs)")
                break
    
    print("\n" + "=" * 60)
    print(f"Training Complete!")
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print("=" * 60)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train')
    ax1.plot(epochs, val_losses, 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train')
    ax2.plot(epochs, val_accs, 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('./checkpoints/training_curves.png', dpi=150)
    print("\n✓ Saved training curves")
    
    # Summary
    print("\nFINAL SUMMARY:")
    print(f"  Initial train acc: {train_accs[0]:.2f}%")
    print(f"  Final train acc:   {train_accs[-1]:.2f}%")
    print(f"  Best val acc:      {best_val_acc:.2f}%")
    
    if best_val_acc < 50:
        print("\n⚠️  WARNING: Accuracy is still low (<50%)")
        print("   Possible issues:")
        print("   1. Data labels are incorrect (run diagnose_data.py)")
        print("   2. Model architecture has bugs")
        print("   3. Images are not properly preprocessed")
        print("   4. LFW dataset structure is wrong")


if __name__ == "__main__":
    main()