"""
Anti-Spoofing — NUAA Correct Split Strategy
============================================
ROOT CAUSE: NUAA has same 14/16 subjects in both real & fake sets.
Subject-aware split still leaks: if subject 0001 is in val-fake,
the model saw their real face in train → recognizes identity, not liveness.

SOLUTION: Split so that for SHARED subjects, both their real AND fake
images go to the same partition. This way the model cannot use
identity as a shortcut — it must learn texture/frequency cues.

Partition strategy:
  - 80% of subjects → train  (both their real + fake images)
  - 20% of subjects → val    (both their real + fake images)
  - Subject 0013 (real only) and 0016 (fake only) assigned carefully
"""

import os
import re
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH     = "./processed_dataset/data.csv"
SAVE_DIR     = "./checkpoints"
IMG_SIZE     = 80
BATCH_SIZE   = 32
NUM_EPOCHS   = 40
LR           = 1e-4
WEIGHT_DECAY = 1e-5
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
FOCAL_ALPHA  = 0.25
FOCAL_GAMMA  = 2.0

# Hand-picked val subjects: choose subjects that have BOTH real+fake,
# spread across the subject ID range for diversity.
# Remaining subjects go to train.
VAL_SUBJECTS = {"0003", "0008", "0015"}   # 3 subjects = ~19% of 16
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Split ─────────────────────────────────────────────────────────────────────

def extract_subject_id(filepath):
    stem = Path(filepath).stem
    m = re.match(r"(\d{4})", stem)
    return m.group(1) if m else Path(filepath).parent.name


def cross_subject_split(df, val_subjects):
    """
    Assign ALL images (real + fake) of a subject to the same partition.
    This eliminates identity leakage entirely.
    """
    df = df.copy()
    df["subject_id"] = df["rgb_path"].apply(extract_subject_id)

    val_df   = df[df["subject_id"].isin(val_subjects)].reset_index(drop=True)
    train_df = df[~df["subject_id"].isin(val_subjects)].reset_index(drop=True)

    print(f"\n  Cross-subject split (identity-leakage-free):")
    print(f"    Train subjects : {sorted(train_df['subject_id'].unique())}")
    print(f"    Val subjects   : {sorted(val_df['subject_id'].unique())}")

    for name, part in [("Train", train_df), ("Val", val_df)]:
        r = (part["label"] == 1).sum()
        f = (part["label"] == 0).sum()
        print(f"    {name}: {len(part)} samples  (real={r}, fake={f})")

    # Verify no subject overlap
    train_subs = set(train_df["subject_id"].unique())
    val_subs   = set(val_df["subject_id"].unique())
    assert len(train_subs & val_subs) == 0, "Subject overlap detected!"
    print(f"    Identity leakage: NONE ✓")

    return train_df, val_df


# ── Dataset ───────────────────────────────────────────────────────────────────

class NUAADataset(Dataset):
    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    VAL_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def __init__(self, df, mode="train"):
        self.df = df.reset_index(drop=True)
        self.transform = (self.TRAIN_TRANSFORMS if mode == "train"
                          else self.VAL_TRANSFORMS)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb = self.transform(Image.open(row["rgb_path"]).convert("RGB"))
        fft = torch.from_numpy(
            np.load(row["fft_path"]).astype(np.float32)
        ).unsqueeze(0)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return rgb, fft, label


def make_loaders(csv_path, val_subjects, batch_size, seed):
    df = pd.read_csv(csv_path)
    df = df[df["rgb_path"].apply(os.path.exists) &
            df["fft_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"  Total valid samples: {len(df)}")

    train_df, val_df = cross_subject_split(df, val_subjects)

    # Weighted sampler — balance real/fake in each train batch
    labels = train_df["label"].values
    class_counts = np.bincount(labels.astype(int))
    weights = 1.0 / class_counts[labels.astype(int)]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                    replacement=True)

    train_ds = NUAADataset(train_df, mode="train")
    val_ds   = NUAADataset(val_df,   mode="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ── Model ─────────────────────────────────────────────────────────────────────

class FFTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class AntiSpoofNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Freeze early layers — only fine-tune last 5 blocks
        for i, layer in enumerate(base.features):
            if i < 14:
                for p in layer.parameters():
                    p.requires_grad = False

        self.rgb_branch = base.features
        self.rgb_pool   = nn.AdaptiveAvgPool2d(1)
        self.fft_branch = FFTBranch()
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 1152, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, rgb, fft):
        r = self.rgb_pool(self.rgb_branch(rgb)).flatten(1)
        f = self.fft_branch(fft)
        return self.classifier(torch.cat([r, f], dim=1)).squeeze(1)


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        at  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (at * (1 - pt) ** self.gamma * bce).mean()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_acer(y_true, y_scores, threshold=0.5):
    preds     = (y_scores >= threshold).astype(int)
    fake_mask = y_true == 0
    real_mask = y_true == 1
    apcer = (preds[fake_mask] == 1).mean() if fake_mask.any() else 0.0
    bpcer = (preds[real_mask] == 0).mean() if real_mask.any() else 0.0
    return apcer, bpcer, (apcer + bpcer) / 2


def find_best_threshold(y_true, y_scores, steps=200):
    best_t, best_acer = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, steps):
        _, _, acer = compute_acer(y_true, y_scores, t)
        if acer < best_acer:
            best_acer, best_t = acer, t
    return best_t, best_acer


# ── Train / Val loops ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for rgb, fft, labels in tqdm(loader, desc="  train", leave=False):
        rgb, fft, labels = rgb.to(device), fft.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(rgb, fft), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_scores = 0.0, [], []
    for rgb, fft, labels in tqdm(loader, desc="  val  ", leave=False):
        rgb, fft, labels = rgb.to(device), fft.to(device), labels.to(device)
        logits = model(rgb, fft)
        total_loss += criterion(logits, labels).item() * len(labels)
        all_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    y_true   = np.array(all_labels)
    y_scores = np.array(all_scores)
    apcer, bpcer, acer = compute_acer(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    return total_loss / len(loader.dataset), apcer, bpcer, acer, auc, y_true, y_scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\nDevice : {DEVICE}")

    print("\nLoading dataset...")
    train_loader, val_loader = make_loaders(
        CSV_PATH, VAL_SUBJECTS, BATCH_SIZE, SEED)

    model     = AntiSpoofNet(dropout=0.5).to(DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS)
    criterion = FocalLoss(FOCAL_ALPHA, FOCAL_GAMMA)

    best_acer, best_epoch = float("inf"), 0
    patience, patience_ctr = 8, 0   # early stopping

    print(f"\nTraining for up to {NUM_EPOCHS} epochs...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | "
          f"{'APCER':>6} | {'BPCER':>6} | {'ACER':>6} | {'AUC':>6}")
    print("-" * 68)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        val_loss, apcer, bpcer, acer, auc, y_true, y_scores = validate(
            model, val_loader, criterion, DEVICE)
        scheduler.step()

        flag = ""
        if acer < best_acer:
            best_acer, best_epoch = acer, epoch
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "acer": acer, "auc": auc,
            }, os.path.join(SAVE_DIR, "best_model_antispoofing.pth"))
            flag = " ← best"
        else:
            patience_ctr += 1

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>8.4f} | "
              f"{apcer:>6.3f} | {bpcer:>6.3f} | {acer:>6.3f} | {auc:>6.3f}{flag}")

        if patience_ctr >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\nBest: epoch {best_epoch}  |  ACER = {best_acer:.4f}")
    ckpt = torch.load(os.path.join(SAVE_DIR, "best_model_antispoofing.pth"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    _, apcer, bpcer, acer, auc, y_true, y_scores = validate(
        model, val_loader, criterion, DEVICE)

    best_t, _ = find_best_threshold(y_true, y_scores)
    apcer_t, bpcer_t, acer_t = compute_acer(y_true, y_scores, best_t)

    print(f"\n── Final Validation Results (cross-subject, no identity leakage) ──")
    print(f"  Default threshold 0.50 → APCER={apcer:.4f}  BPCER={bpcer:.4f}  ACER={acer:.4f}")
    print(f"  Best threshold  {best_t:.2f}  → APCER={apcer_t:.4f}  BPCER={bpcer_t:.4f}  ACER={acer_t:.4f}")
    print(f"  AUC-ROC : {auc:.4f}\n")
    print(classification_report(
        y_true, (y_scores >= best_t).astype(int),
        target_names=["fake", "real"]))

    # Save threshold with model
    torch.save({
        "epoch":       best_epoch,
        "model_state": model.state_dict(),
        "acer":        acer_t,
        "auc":         auc,
        "threshold":   float(best_t),
    }, os.path.join(SAVE_DIR, "best_model_antispoofing.pth"))
    print(f"  Threshold {best_t:.4f} saved to best_model_antispoofing.pth")


if __name__ == "__main__":
    main()