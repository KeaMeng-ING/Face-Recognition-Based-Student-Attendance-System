"""
Face recognition training (project-native, no LFW dependency).

Trains MobileFaceNet embeddings with an ArcFace-style classification head
on an ImageFolder dataset (default: dataset/vggface2_train).

Output checkpoint is compatible with src/app.py:
  checkpoints/best_model.pth  -> contains key "model_state_dict"

Usage:
  python src/train.py
  python src/train.py --data-dir dataset/vggface2_train --epochs 30
"""

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from mobilefacenet import MobileFaceNet


@dataclass
class Config:
    data_dir: str = "./dataset/vggface2_train"
    save_path: str = "./checkpoints/best_model.pth"
    embedding_size: int = 512
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_split: float = 0.2
    num_workers: int = 2
    arcface_s: float = 30.0
    arcface_m: float = 0.30
    max_images: int = 0
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ArcFaceHead(nn.Module):
    """ArcFace classification head."""

    def __init__(self, embedding_size: int, num_classes: int, s: float, m: float):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(embeddings, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(emb, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
        return logits


@torch.no_grad()
def compute_acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).float().mean().item() * 100.0


def make_dataloaders(cfg: Config, device: torch.device):
    tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = datasets.ImageFolder(cfg.data_dir, transform=tf)
    if cfg.max_images > 0 and cfg.max_images < len(dataset):
        idx = np.random.default_rng(cfg.seed).choice(len(dataset), cfg.max_images, replace=False)
        dataset = Subset(dataset, sorted(idx.tolist()))

    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )

    class_count = len(dataset.dataset.classes) if isinstance(dataset, Subset) else len(dataset.classes)
    class_to_idx = dataset.dataset.class_to_idx if isinstance(dataset, Subset) else dataset.class_to_idx

    return train_loader, val_loader, class_count, class_to_idx, len(train_ds), len(val_ds)


def train_one_epoch(model, head, loader, optimizer, device):
    model.train()
    head.train()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        logits = head(embeddings, labels)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), max_norm=5.0)
        optimizer.step()

        bs = labels.size(0)
        total_samples += bs
        total_loss += loss.item() * bs
        total_acc += compute_acc(logits.detach(), labels) * bs

    return total_loss / max(total_samples, 1), total_acc / max(total_samples, 1)


@torch.no_grad()
def validate(model, head, loader, device):
    model.eval()
    head.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        logits = head(embeddings, labels)
        loss = F.cross_entropy(logits, labels)

        bs = labels.size(0)
        total_samples += bs
        total_loss += loss.item() * bs
        total_acc += compute_acc(logits, labels) * bs

    return total_loss / max(total_samples, 1), total_acc / max(total_samples, 1)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train MobileFaceNet on project dataset")
    parser.add_argument("--data-dir", default=Config.data_dir)
    parser.add_argument("--save-path", default=Config.save_path)
    parser.add_argument("--embedding-size", type=int, default=Config.embedding_size)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--val-split", type=float, default=Config.val_split)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--arcface-s", type=float, default=Config.arcface_s)
    parser.add_argument("--arcface-m", type=float, default=Config.arcface_m)
    parser.add_argument("--max-images", type=int, default=Config.max_images)
    parser.add_argument("--seed", type=int, default=Config.seed)
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    if not os.path.isdir(cfg.data_dir):
        raise FileNotFoundError(f"Dataset not found: {cfg.data_dir}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader, class_count, class_to_idx, train_n, val_n = make_dataloaders(cfg, device)

    model = MobileFaceNet(embedding_size=cfg.embedding_size).to(device)
    head = ArcFaceHead(
        embedding_size=cfg.embedding_size,
        num_classes=class_count,
        s=cfg.arcface_s,
        m=cfg.arcface_m,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)

    best_val_acc = -1.0
    best_epoch = 0

    print("=" * 72)
    print("Face Recognition Training (no LFW)")
    print("=" * 72)
    print(f"Device        : {device}")
    print(f"Dataset       : {cfg.data_dir}")
    print(f"Classes       : {class_count}")
    print(f"Train samples : {train_n}")
    print(f"Val samples   : {val_n}")
    print(f"Save path     : {cfg.save_path}")
    print("-" * 72)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 72)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, head, train_loader, optimizer, device)
        va_loss, va_acc = validate(model, head, val_loader, device)
        scheduler.step()

        mark = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_to_idx": class_to_idx,
                    "embedding_size": cfg.embedding_size,
                    "arcface_s": cfg.arcface_s,
                    "arcface_m": cfg.arcface_m,
                },
                cfg.save_path,
            )
            mark = " <-- best"

        print(
            f"{epoch:5d} | {tr_loss:10.4f} | {tr_acc:9.2f} | {va_loss:8.4f} | {va_acc:7.2f}{mark}"
        )

    print("-" * 72)
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Checkpoint saved to: {cfg.save_path}")


if __name__ == "__main__":
    main()
