"""
Complete Training Script for Face Recognition
MobileFaceNet + ArcFace Loss on LFW Subset
Ready to run!
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from mobilefacenet import MobileFaceNet
from arcface_loss import ArcFaceLoss  # We'll create this next


# ============================================================================
# Dataset Class
# ============================================================================

class LFWDataset(Dataset):
    """
    Dataset for LFW subset with multiple images per person
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# Prepare LFW Dataset
# ============================================================================

def prepare_lfw_data(lfw_root="./lfw", min_images=10, train_ratio=0.7, val_ratio=0.15):
    """
    Filter and split LFW dataset
    
    Returns:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, num_classes
    """
    from sklearn.model_selection import train_test_split
    
    print("Scanning LFW directory...")
    
    all_paths = []
    all_labels = []
    label_to_idx = {}
    current_idx = 0
    
    # Scan directory and filter
    for person_name in os.listdir(lfw_root):
        person_dir = os.path.join(lfw_root, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        images = [os.path.join(person_dir, img) for img in os.listdir(person_dir) 
                  if img.endswith(('.jpg', '.png'))]
        
        # Only keep people with enough images
        if len(images) >= min_images:
            label_to_idx[person_name] = current_idx
            
            for img_path in images:
                all_paths.append(img_path)
                all_labels.append(current_idx)
            
            current_idx += 1
    
    num_classes = len(label_to_idx)
    print(f"Found {num_classes} valid identities with {len(all_paths)} total images")
    
    # Convert to numpy for sklearn
    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)
    
    # Split: 70% train, 15% val, 15% test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=(1-train_ratio), stratify=all_labels, random_state=42
    )
    
    val_ratio_adjusted = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1-val_ratio_adjusted), 
        stratify=temp_labels, random_state=42
    )
    
    print(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    
    return (train_paths, train_labels, 
            val_paths, val_labels, 
            test_paths, test_labels, 
            num_classes, label_to_idx)


# ============================================================================
# Data Transforms
# ============================================================================

def get_transforms():
    """Get training and validation transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(model, criterion, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    criterion.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # For accuracy, we need to check which class has highest score
        with torch.no_grad():
            # Get the logits from ArcFace (before softmax)
            cosine = torch.mm(
                torch.nn.functional.normalize(embeddings), 
                torch.nn.functional.normalize(criterion.weight).t()
            )
            predictions = torch.argmax(cosine, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, criterion, val_loader, device):
    """Validate the model"""
    model.eval()
    criterion.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            
            running_loss += loss.item()
            
            # Accuracy
            cosine = torch.mm(
                torch.nn.functional.normalize(embeddings), 
                torch.nn.functional.normalize(criterion.weight).t()
            )
            predictions = torch.argmax(cosine, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / len(val_loader), 100. * correct / total


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Configuration
    config = {
        'lfw_root': './lfw',
        'batch_size': 64,
        'num_epochs': 30,
        'learning_rate': 0.1,
        'embedding_size': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'save_dir': './checkpoints'
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print("=" * 60)
    print("Face Recognition Training - MobileFaceNet + ArcFace")
    print("=" * 60)
    print(f"Device: {config['device']}")
    
    # 1. Prepare data
    (train_paths, train_labels, 
     val_paths, val_labels, 
     test_paths, test_labels, 
     num_classes, label_to_idx) = prepare_lfw_data(config['lfw_root'])
    
    # 2. Create datasets
    train_transform, val_transform = get_transforms()
    
    train_dataset = LFWDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = LFWDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 3. Create model
    model = MobileFaceNet(embedding_size=config['embedding_size'])
    model = model.to(config['device'])
    
    # 4. Create ArcFace loss
    criterion = ArcFaceLoss(
        embedding_size=config['embedding_size'],
        num_classes=num_classes,
        s=64.0,
        m=0.5
    )
    criterion = criterion.to(config['device'])
    
    # 5. Setup optimizer
    optimizer = optim.SGD(
        [{'params': model.parameters()}, {'params': criterion.parameters()}],
        lr=config['learning_rate'],
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # 6. Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 20, 25],
        gamma=0.1
    )
    
    # 7. Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, criterion, train_loader, optimizer, config['device'], epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, criterion, val_loader, config['device'])
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_to_idx': label_to_idx
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    # 8. Plot training curves
    plot_training_curves(history, config['save_dir'])
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 60)


def plot_training_curves(history, save_dir):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    print(f"✓ Saved training curves to {save_dir}/training_curves.png")


if __name__ == "__main__":
    main()