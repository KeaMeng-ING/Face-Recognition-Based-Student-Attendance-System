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
CSV_PATH     = "./dataset/processed_dataset/data.csv"
SAVE_DIR     = "./checkpoints"
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 60
LR           = 1e-4
WEIGHT_DECAY = 1e-5
SEED         = 42
DEVICE       = "mps" if torch.backends.mps.is_available() else \
               "cuda" if torch.cuda.is_available() else "cpu"
FOCAL_ALPHA  = 0.25
FOCAL_GAMMA  = 3.0

# Hand-picked val subjects: choose subjects that have BOTH real+fake,
# spread across the subject ID range for diversity.
# Remaining subjects go to train.
VAL_SUBJECTS = {"0001", "0003", "0008", "0015"}  # 25% instead of 19%
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Split ─────────────────────────────────────────────────────────────────────

def extract_subject_id(filepath):
    """
    Extract 4-digit NUAA subject ID from filename (e.g. printed_photo_0001_xxx → '0001').
    For non-NUAA files (screen attacks), returns the attack_type prefix as the group ID
    so they get their own consistent split bucket.
    """
    stem = Path(filepath).stem
    # Search anywhere in the stem for a 4-digit subject ID
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", stem)
    if m:
        return m.group(1)
    # No subject ID — use attack type prefix (everything before last _NNNNN)
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem


def cross_subject_split(df, val_subjects):
    """
    For NUAA data: assign all images of a subject to the same partition
    (identity-leakage-free cross-subject split).

    For non-NUAA data (screen attacks): random 80/20 split per attack type
    so each attack type is represented in both train and val.
    """
    df = df.copy()
    df["subject_id"] = df["rgb_path"].apply(extract_subject_id)

    # NUAA rows have 4-digit subject IDs like '0001'
    nuaa_mask    = df["subject_id"].str.match(r"^\d{4}$")
    nuaa_df      = df[nuaa_mask]
    non_nuaa_df  = df[~nuaa_mask]

    # NUAA: cross-subject split
    nuaa_val   = nuaa_df[nuaa_df["subject_id"].isin(val_subjects)]
    nuaa_train = nuaa_df[~nuaa_df["subject_id"].isin(val_subjects)]

    # Non-NUAA: random 20% val per attack type
    non_nuaa_train_parts, non_nuaa_val_parts = [], []
    for attack_type, group in non_nuaa_df.groupby("subject_id"):
        group = group.sample(frac=1, random_state=SEED)
        split = max(1, int(len(group) * 0.2))
        non_nuaa_val_parts.append(group.iloc[:split])
        non_nuaa_train_parts.append(group.iloc[split:])

    non_nuaa_val   = pd.concat(non_nuaa_val_parts)   if non_nuaa_val_parts   else pd.DataFrame()
    non_nuaa_train = pd.concat(non_nuaa_train_parts) if non_nuaa_train_parts else pd.DataFrame()

    train_df = pd.concat([nuaa_train, non_nuaa_train]).reset_index(drop=True)
    val_df   = pd.concat([nuaa_val,   non_nuaa_val  ]).reset_index(drop=True)

    print(f"\n  Split summary:")
    print(f"    NUAA train subjects : {sorted(nuaa_train['subject_id'].unique())}")
    print(f"    NUAA val subjects   : {sorted(nuaa_val['subject_id'].unique())}")
    if len(non_nuaa_df) > 0:
        print(f"    Screen attack types (80/20 random split):")
        for at, g in non_nuaa_df.groupby("subject_id"):
            in_val = len(non_nuaa_val[non_nuaa_val["subject_id"] == at])
            print(f"      {at:<30} train={len(g)-in_val}  val={in_val}")

    for name, part in [("Train", train_df), ("Val", val_df)]:
        r = (part["label"] == 1).sum()
        f = (part["label"] == 0).sum()
        print(f"    {name}: {len(part)} samples  (real={r}, fake={f})")

    return train_df, val_df


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_fft(path):
    """Load FFT with normalization to improve training stability."""
    arr = np.load(path).astype(np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return torch.from_numpy(arr).unsqueeze(0)


class NUAADataset(Dataset):
    TRAIN_TRANSFORMS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
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
        fft = load_fft(row["fft_path"])
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return rgb, fft, label


def make_loaders(csv_path, val_subjects, batch_size, seed):
    df = pd.read_csv(csv_path)
    df = df[df["rgb_path"].apply(os.path.exists) &
            df["fft_path"].apply(os.path.exists)].reset_index(drop=True)
    
    # Calculate sample weights: give 10x more importance to non-NUAA (screen) attacks
    # as they are currently being ignored by the model.
    df["is_personal"] = df["rgb_path"].str.contains("my_spoof_data")
    
    print(f"  Total valid samples: {len(df)}")
    print(f"  Personal samples: {df['is_personal'].sum()}")

    train_df, val_df = cross_subject_split(df, val_subjects)

    # Weighted sampler — balance classes AND prioritize personal data
    weights = []
    for _, row in train_df.iterrows():
        # Base weight by class frequency
        base_w = 1.0 if row["label"] == 1 else 0.5 # prioritize real slightly to avoid spoof-paranoia
        # Multiplier for personal data to force the model to learn it
        personal_multiplier = 15.0 if "my_spoof_data" in row["rgb_path"] else 1.0
        weights.append(base_w * personal_multiplier)
    
    sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                    replacement=True)

    train_ds = NUAADataset(train_df, mode="train")
    val_ds   = NUAADataset(val_df,   mode="val")

    pin = DEVICE == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=pin)
    return train_loader, val_loader


# ── Model ─────────────────────────────────────────────────────────────────────

class FFTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            # AvgPool2d(3,3) on 10x10 input → 3x3 output (same as AdaptiveAvgPool2d((3,3)))
            # AdaptiveAvgPool2d is not supported on MPS when sizes aren't divisible
            nn.AvgPool2d(kernel_size=3, stride=3),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class AntiSpoofNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Freeze early layers — fine-tune from block 6 onwards
        for i, layer in enumerate(base.features):
            if i < 6:
                for p in layer.parameters():
                    p.requires_grad = False

        self.rgb_branch = base.features
        self.rgb_pool   = nn.AdaptiveAvgPool2d(1)
        self.fft_branch = FFTBranch()
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 2304, 512),
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