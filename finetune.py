"""
Anti-Spoofing Fine-Tuner
=========================
Fine-tunes your existing NUAA-trained model on your own
collected real/fake samples from collect_data.py.

Strategy:
  - Freeze ALL layers except the classifier head
  - Train only the head on your personal data (fast, few samples needed)
  - Low LR to avoid forgetting NUAA knowledge (catastrophic forgetting prevention)
  - Saves as best_model_antispoofing.pth (replaces old model in-place)

Usage:
    python finetune.py

Requirements:
    Run collect_data.py first to gather your personal samples.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────
MY_DATA_DIR    = "./my_spoof_data"      # output of collect_data.py
BASE_MODEL     = "./checkpoints/best_model_antispoofing.pth"
SAVE_PATH      = "./checkpoints/best_model_antispoofing.pth"  # overwrite in-place
IMG_SIZE       = 80
BATCH_SIZE     = 16                     # small batch — personal dataset is small
NUM_EPOCHS     = 30
LR             = 5e-5                   # very low — only nudging the head
WEIGHT_DECAY   = 1e-4
SEED           = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Focal loss params
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(seed):
    import random; random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model (same architecture as training) ─────────────────────────────────────

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
    def forward(self, x): return self.net(x).flatten(1)


class AntiSpoofNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.rgb_branch = base.features
        self.rgb_pool   = nn.AdaptiveAvgPool2d(1)
        self.fft_branch = FFTBranch()
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 1152, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, rgb, fft):
        r = self.rgb_pool(self.rgb_branch(rgb)).flatten(1)
        f = self.fft_branch(fft)
        return self.classifier(torch.cat([r, f], dim=1)).squeeze(1)


# ── Dataset ───────────────────────────────────────────────────────────────────

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    # Strong augmentation to simulate varied conditions
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


def compute_fft(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    mag  = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)
    mag  = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return torch.from_numpy(mag.astype(np.float32)).unsqueeze(0)


class PersonalDataset(Dataset):
    def __init__(self, records, mode="train"):
        self.records   = records
        self.transform = TRAIN_TRANSFORMS if mode == "train" else VAL_TRANSFORMS

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = cv2.imread(path)
        if img is None:
            # Return blank sample on read failure
            rgb = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            fft = torch.zeros(1, IMG_SIZE, IMG_SIZE)
            return rgb, fft, torch.tensor(float(label))

        # Apply CLAHE (same as attendance.py inference path)
        lab    = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l,a,b  = cv2.split(lab)
        clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        l      = clahe.apply(l)
        img    = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

        rgb = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fft = compute_fft(img)
        return rgb, fft, torch.tensor(float(label))


def load_records(data_dir):
    records = []
    for label, subdir in [(1, "real"), (0, "fake")]:
        d = os.path.join(data_dir, subdir)
        if not os.path.exists(d):
            continue
        for f in Path(d).rglob("*.jpg"):
            records.append((str(f), label))
    return records


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        at  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (at * (1 - pt) ** self.gamma * bce).mean()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_acer(y_true, y_scores, threshold=0.5):
    preds = (y_scores >= threshold).astype(int)
    fake_mask = y_true == 0; real_mask = y_true == 1
    apcer = (preds[fake_mask] == 1).mean() if fake_mask.any() else 0.0
    bpcer = (preds[real_mask] == 0).mean() if real_mask.any() else 0.0
    return apcer, bpcer, (apcer + bpcer) / 2


def find_best_threshold(y_true, y_scores):
    best_t, best_acer = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 200):
        _, _, acer = compute_acer(y_true, y_scores, t)
        if acer < best_acer:
            best_acer, best_t = acer, t
    return best_t, best_acer


# ── Train / Val ───────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for rgb, fft, labels in tqdm(loader, desc="  train", leave=False):
        rgb, fft, labels = rgb.to(DEVICE), fft.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(rgb, fft), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total, all_labels, all_scores = 0.0, [], []
    for rgb, fft, labels in tqdm(loader, desc="  val  ", leave=False):
        rgb, fft, labels = rgb.to(DEVICE), fft.to(DEVICE), labels.to(DEVICE)
        logits = model(rgb, fft)
        total += criterion(logits, labels).item() * len(labels)
        all_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    y_true  = np.array(all_labels)
    y_scores = np.array(all_scores)
    _, _, acer = compute_acer(y_true, y_scores)
    return total / len(loader.dataset), acer, y_true, y_scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    print(f"\nDevice : {DEVICE}")

    # ── Load personal data ────────────────────────────────────────────────────
    records = load_records(MY_DATA_DIR)
    if not records:
        print(f"❌ No data found in {MY_DATA_DIR}")
        print("   Run collect_data.py first.")
        return

    real_n = sum(1 for _,l in records if l==1)
    fake_n = sum(1 for _,l in records if l==0)
    print(f"\nPersonal dataset: {len(records)} samples  (real={real_n}, fake={fake_n})")

    if real_n < 50 or fake_n < 50:
        print("⚠️  Less than 50 samples per class — collect more for better results.")

    train_rec, val_rec = train_test_split(
        records, test_size=0.2, random_state=SEED,
        stratify=[l for _,l in records])

    # Weighted sampler
    train_labels  = np.array([l for _,l in train_rec])
    class_counts  = np.bincount(train_labels)
    sample_weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_ds = PersonalDataset(train_rec, mode="train")
    val_ds   = PersonalDataset(val_rec,   mode="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

    # ── Load base model ───────────────────────────────────────────────────────
    model = AntiSpoofNet().to(DEVICE)
    ckpt  = torch.load(BASE_MODEL, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    old_threshold = float(ckpt.get("threshold", 0.43))
    print(f"Base model loaded  (old threshold={old_threshold:.4f})")

    # ── Freeze strategy ───────────────────────────────────────────────────────
    # Freeze everything except the classifier head.
    # The backbone already knows texture/FFT features from NUAA —
    # we only teach the head new decision boundaries for your camera.
    for param in model.rgb_branch.parameters():
        param.requires_grad = False
    for param in model.fft_branch.parameters():
        param.requires_grad = False
    # Classifier head stays trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (head only): {trainable:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS)
    criterion = FocalLoss(FOCAL_ALPHA, FOCAL_GAMMA)

    best_acer, best_epoch = float("inf"), 0
    patience_ctr, patience = 0, 10

    print(f"\nFine-tuning for up to {NUM_EPOCHS} epochs on your personal data...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'ACER':>6}")
    print("-" * 42)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, acer, y_true, y_scores = validate(model, val_loader, criterion)
        scheduler.step()

        flag = ""
        if acer < best_acer:
            best_acer, best_epoch = acer, epoch
            patience_ctr = 0
            # Find best threshold on val set
            best_t, _ = find_best_threshold(y_true, y_scores)
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "acer":        acer,
                "threshold":   float(best_t),
                "finetuned":   True,
            }, SAVE_PATH)
            flag = f" ← best (thr={best_t:.4f})"
        else:
            patience_ctr += 1

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>8.4f} | {acer:>6.4f}{flag}")

        if patience_ctr >= patience:
            print(f"\nEarly stop at epoch {epoch}")
            break

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\nBest: epoch {best_epoch}  ACER={best_acer:.4f}")

    ckpt = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, _, y_true, y_scores = validate(model, val_loader, criterion)

    best_t, _ = find_best_threshold(y_true, y_scores)
    apcer, bpcer, acer = compute_acer(y_true, y_scores, best_t)

    print(f"\n── Fine-tuned Model Results ────────────────────────────────")
    print(f"  APCER (fake→real) : {apcer:.4f}")
    print(f"  BPCER (real→fake) : {bpcer:.4f}")
    print(f"  ACER              : {acer:.4f}")
    print(f"  New threshold     : {best_t:.4f}  (was {old_threshold:.4f})")
    print()
    print(classification_report(y_true, (y_scores>=best_t).astype(int),
                                 target_names=["fake","real"]))

    # Save with updated threshold
    torch.save({
        "epoch":       best_epoch,
        "model_state": model.state_dict(),
        "acer":        acer,
        "threshold":   float(best_t),
        "finetuned":   True,
    }, SAVE_PATH)
    print(f"✅ Saved to {SAVE_PATH}  (threshold={best_t:.4f})")
    print("\nRestart attendance.py — it will automatically use the updated model.")


if __name__ == "__main__":
    main()
