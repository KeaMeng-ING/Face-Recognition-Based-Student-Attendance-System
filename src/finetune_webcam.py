"""
finetune_webcam.py
==================
Fine-tunes the existing AntiSpoofNet checkpoint on your webcam data.
Keeps NUAA data in the mix so the model doesn't forget printed-photo attacks.

Requirements:
  - Collect webcam data first:
      python collect_webcam_data.py --label real --n 300
      python collect_webcam_data.py --label fake --n 300
  - Existing checkpoint at ./checkpoints/best_model_antispoofing.pth

Usage:
    python finetune_webcam.py
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm
from PIL import Image
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
WEBCAM_REAL_DIR  = "./dataset/webcam_data/real"
WEBCAM_FAKE_DIR  = "./dataset/webcam_data/fake"
NUAA_CSV         = "./dataset/processed_dataset/data.csv"   # set to None to skip NUAA
CHECKPOINT_IN    = "./checkpoints/best_model_antispoofing.pth"
CHECKPOINT_OUT   = "./checkpoints/best_model_antispoofing.pth"  # overwrites in place

IMG_SIZE     = 224
BATCH_SIZE   = 16
NUM_EPOCHS   = 30
LR           = 5e-5       # small LR — fine-tuning, not training from scratch
WEIGHT_DECAY = 1e-5
SEED         = 42
VAL_SPLIT    = 0.20       # 20% of webcam data → validation
WEBCAM_WEIGHT = 8.0       # how much more to weight webcam vs NUAA samples
DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


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
            nn.AvgPool2d(kernel_size=3, stride=3),
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
            nn.Linear(1280 + 2304, 512), nn.BatchNorm1d(512), nn.ReLU(),
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

TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SpoofDataset(Dataset):
    def __init__(self, samples, mode="train"):
        """
        samples: list of (rgb_path, fft_path, label)
        label: 1=real, 0=fake
        """
        self.samples   = samples
        self.transform = TRAIN_TF if mode == "train" else VAL_TF

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, fft_path, label = self.samples[idx]
        rgb = self.transform(Image.open(rgb_path).convert("RGB"))
        arr = np.load(fft_path).astype(np.float32)
        # Re-normalize FFT just in case (already done at collection time)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        fft = torch.from_numpy(arr).unsqueeze(0)
        return rgb, fft, torch.tensor(float(label), dtype=torch.float32)


# ── Build sample list ─────────────────────────────────────────────────────────

def load_webcam_samples():
    samples = []
    for label_int, folder in [(1, WEBCAM_REAL_DIR), (0, WEBCAM_FAKE_DIR)]:
        jpgs = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        for rgb_path in jpgs:
            fft_path = rgb_path.replace(".jpg", "_fft.npy")
            if os.path.exists(fft_path):
                samples.append((rgb_path, fft_path, label_int))
            else:
                print(f"  ⚠️  Missing FFT for {rgb_path} — skipped")
    return samples


def load_nuaa_samples():
    if NUAA_CSV is None or not os.path.exists(NUAA_CSV):
        return []
    df = pd.read_csv(NUAA_CSV)
    df = df[df["rgb_path"].apply(os.path.exists) &
            df["fft_path"].apply(os.path.exists)]
    return list(zip(df["rgb_path"], df["fft_path"], df["label"]))


def split_samples(samples, val_frac=0.20, seed=42):
    random.seed(seed)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * val_frac))
    return samples[n_val:], samples[:n_val]


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
    preds     = (y_scores >= threshold).astype(int)
    fake_mask = y_true == 0; real_mask = y_true == 1
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


# ── Train / Val ───────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for rgb, fft, labels in tqdm(loader, desc="  train", leave=False):
        rgb, fft, labels = rgb.to(device), fft.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(rgb, fft), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total, all_labels, all_scores = 0.0, [], []
    for rgb, fft, labels in tqdm(loader, desc="  val  ", leave=False):
        rgb, fft, labels = rgb.to(device), fft.to(device), labels.to(device)
        logits = model(rgb, fft)
        total += criterion(logits, labels).item() * len(labels)
        all_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    y_true   = np.array(all_labels)
    y_scores = np.array(all_scores)
    apcer, bpcer, acer = compute_acer(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    return total / len(loader.dataset), apcer, bpcer, acer, auc, y_true, y_scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    print(f"\nDevice: {DEVICE}")

    # ── Load samples ──────────────────────────────────────────────────────────
    webcam_samples = load_webcam_samples()
    nuaa_samples   = load_nuaa_samples()

    if len(webcam_samples) == 0:
        print("\n❌ No webcam samples found.")
        print("   Run: python collect_webcam_data.py --label real --n 300")
        print("   Run: python collect_webcam_data.py --label fake --n 300")
        return

    real_count = sum(1 for _, _, l in webcam_samples if l == 1)
    fake_count = sum(1 for _, _, l in webcam_samples if l == 0)
    print(f"\nWebcam samples: {len(webcam_samples)}  (real={real_count}, fake={fake_count})")
    print(f"NUAA samples  : {len(nuaa_samples)}")

    if fake_count == 0:
        print("\n⚠️  No fake webcam samples found!")
        print("   Run: python collect_webcam_data.py --label fake --n 300")
        print("   Continuing with NUAA fake data only — results may be poor.")

    # ── Split webcam 80/20 ────────────────────────────────────────────────────
    webcam_train, webcam_val = split_samples(webcam_samples, VAL_SPLIT, SEED)

    # NUAA: use all for training (already validated on its own split during orig training)
    # Use a subset so webcam dominates — take at most 2x webcam size from NUAA
    max_nuaa = len(webcam_train) * 2
    if len(nuaa_samples) > max_nuaa:
        random.shuffle(nuaa_samples)
        nuaa_samples = nuaa_samples[:max_nuaa]

    train_samples = webcam_train + nuaa_samples
    val_samples   = webcam_val   # validate ONLY on webcam — that's what matters

    print(f"\nTrain: {len(train_samples)}  (webcam={len(webcam_train)}, nuaa={len(nuaa_samples)})")
    print(f"Val  : {len(val_samples)}  (webcam only)")

    # ── Weighted sampler: heavily favour webcam samples ───────────────────────
    weights = []
    for rgb_path, _, label in train_samples:
        is_webcam = "webcam_data" in rgb_path
        w = WEBCAM_WEIGHT if is_webcam else 1.0
        weights.append(w)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = SpoofDataset(train_samples, mode="train")
    val_ds   = SpoofDataset(val_samples,   mode="val")
    pin      = DEVICE == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)

    # ── Load existing checkpoint ───────────────────────────────────────────────
    print(f"\nLoading checkpoint: {CHECKPOINT_IN}")
    ckpt  = torch.load(CHECKPOINT_IN, map_location=DEVICE, weights_only=False)
    model = AntiSpoofNet().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Original ACER: {ckpt.get('acer', '?')}   threshold: {ckpt.get('threshold', '?')}")

    # Freeze RGB branch early layers — only fine-tune classifier + FFT branch + top RGB layers
    for i, layer in enumerate(model.rgb_branch):
        for p in layer.parameters():
            p.requires_grad = (i >= 12)   # unfreeze last 6 MobileNetV2 blocks

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_acer, best_epoch = float("inf"), 0
    patience_ctr, PATIENCE = 0, 8
    best_state = None

    print(f"\nFine-tuning for up to {NUM_EPOCHS} epochs...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | "
          f"{'APCER':>6} | {'BPCER':>6} | {'ACER':>6} | {'AUC':>6}")
    print("-" * 68)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, apcer, bpcer, acer, auc, y_true, y_scores = validate(
            model, val_loader, criterion, DEVICE)
        scheduler.step()

        flag = ""
        if acer < best_acer:
            best_acer, best_epoch = acer, epoch
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            flag = " ← best"
        else:
            patience_ctr += 1

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>8.4f} | "
              f"{apcer:>6.3f} | {bpcer:>6.3f} | {acer:>6.3f} | {auc:>6.3f}{flag}")

        if patience_ctr >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── Save ──────────────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(DEVICE)

    _, apcer, bpcer, acer, auc, y_true, y_scores = validate(
        model, val_loader, criterion, DEVICE)
    best_t, _ = find_best_threshold(y_true, y_scores)
    apcer_t, bpcer_t, acer_t = compute_acer(y_true, y_scores, best_t)

    print(f"\n── Final Results (webcam val set) ──")
    print(f"  Threshold 0.50 → APCER={apcer:.4f}  BPCER={bpcer:.4f}  ACER={acer:.4f}")
    print(f"  Best threshold {best_t:.2f} → APCER={apcer_t:.4f}  BPCER={bpcer_t:.4f}  ACER={acer_t:.4f}")
    print(f"  AUC: {auc:.4f}")
    print()
    print(classification_report(
        y_true, (y_scores >= best_t).astype(int),
        target_names=["fake", "real"]))

    torch.save({
        "epoch":       best_epoch,
        "model_state": best_state,
        "acer":        acer_t,
        "auc":         auc,
        "threshold":   float(best_t),
    }, CHECKPOINT_OUT)
    print(f"\n✅ Saved to {CHECKPOINT_OUT}  (threshold={best_t:.4f})")
    print("   Restart app.py — it will load the new model automatically.")


if __name__ == "__main__":
    main()
