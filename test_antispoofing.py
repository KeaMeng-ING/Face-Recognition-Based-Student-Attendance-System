"""
Anti-Spoofing Model — Test / Evaluation Script
===============================================
Usage:
    # Basic evaluation against a CSV
    python test_antispoofing.py --csv ./processed_dataset/data.csv --checkpoint ./checkpoints/best_model_antispoofing.pth

    # Evaluate only on specific subjects (e.g. held-out test subjects)
    python test_antispoofing.py --csv ./processed_dataset/data.csv --checkpoint ./checkpoints/best_model_antispoofing.pth --subjects 0003 0008 0015

    # Override threshold (instead of using saved one)
    python test_antispoofing.py --csv ./processed_dataset/data.csv --checkpoint ./checkpoints/best_model_antispoofing.pth --threshold 0.45

    # Save per-sample predictions to CSV
    python test_antispoofing.py --csv ./processed_dataset/data.csv --checkpoint ./checkpoints/best_model_antispoofing.pth --save-preds ./predictions.csv

    # Test on a single image + fft pair
    python test_antispoofing.py --single-rgb ./path/to/face.jpg --single-fft ./path/to/face_fft.npy --checkpoint ./checkpoints/best_model_antispoofing.pth
"""

import argparse
import os
import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, roc_curve
)
from tqdm import tqdm
from PIL import Image


# ── Model (must match training exactly) ───────────────────────────────────────

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
        self.rgb_branch = base.features
        self.rgb_pool   = nn.AdaptiveAvgPool2d(1)
        self.fft_branch = FFTBranch()
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 1152, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, rgb, fft):
        r = self.rgb_pool(self.rgb_branch(rgb)).flatten(1)
        f = self.fft_branch(fft)
        return self.classifier(torch.cat([r, f], dim=1)).squeeze(1)


# ── Data helpers ───────────────────────────────────────────────────────────────

IMG_SIZE = 80

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_fft(path: str) -> torch.Tensor:
    """Load raw FFT without normalization (matches training script)."""
    arr = np.load(path).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def extract_subject_id(filepath: str) -> str:
    m = re.match(r"(\d{4})", Path(filepath).stem)
    return m.group(1) if m else Path(filepath).parent.name


class EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb = VAL_TRANSFORMS(Image.open(row["rgb_path"]).convert("RGB"))
        fft = load_fft(row["fft_path"])
        label = int(row["label"])
        subject = row.get("subject_id", "unknown")
        return rgb, fft, label, str(subject), str(row["rgb_path"])


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_acer(y_true, y_scores, threshold=0.5):
    preds     = (y_scores >= threshold).astype(int)
    fake_mask = y_true == 0
    real_mask = y_true == 1
    apcer = (preds[fake_mask] == 1).mean() if fake_mask.any() else 0.0
    bpcer = (preds[real_mask] == 0).mean() if real_mask.any() else 0.0
    return float(apcer), float(bpcer), float((apcer + bpcer) / 2)


def find_best_threshold(y_true, y_scores, steps=500):
    best_t, best_acer = 0.5, float("inf")
    for t in np.linspace(0.01, 0.99, steps):
        _, _, acer = compute_acer(y_true, y_scores, t)
        if acer < best_acer:
            best_acer, best_t = acer, t
    return float(best_t), float(best_acer)


def eer_from_roc(y_true, y_scores):
    """Equal Error Rate: point where FPR ≈ FNR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2), float(thresholds[idx])


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_scores, all_labels, all_subjects, all_paths = [], [], [], []

    for rgb, fft, labels, subjects, paths in tqdm(loader, desc="Evaluating"):
        rgb  = rgb.to(device)
        fft  = fft.to(device)
        logits = model(rgb, fft)
        scores = torch.sigmoid(logits).cpu().numpy().tolist()
        all_scores.extend(scores)
        all_labels.extend(labels.numpy().tolist())
        all_subjects.extend(subjects)
        all_paths.extend(paths)

    return (np.array(all_labels), np.array(all_scores),
            np.array(all_subjects), np.array(all_paths))


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def full_report(y_true, y_scores, threshold, args):
    apcer, bpcer, acer = compute_acer(y_true, y_scores, threshold)
    auc   = roc_auc_score(y_true, y_scores)
    eer, eer_t = eer_from_roc(y_true, y_scores)
    preds = (y_scores >= threshold).astype(int)
    cm    = confusion_matrix(y_true, preds)

    print_section("Overall Results")
    print(f"  Samples evaluated : {len(y_true)}")
    print(f"  Real (label=1)    : {(y_true == 1).sum()}")
    print(f"  Fake (label=0)    : {(y_true == 0).sum()}")
    print(f"\n  Threshold used    : {threshold:.4f}")
    print(f"  APCER (fake→real) : {apcer:.4f}  ({apcer*100:.1f}%)")
    print(f"  BPCER (real→fake) : {bpcer:.4f}  ({bpcer*100:.1f}%)")
    print(f"  ACER              : {acer:.4f}  ({acer*100:.1f}%)")
    print(f"  AUC-ROC           : {auc:.4f}")
    print(f"  EER               : {eer:.4f}  (at threshold {eer_t:.4f})")

    print_section("Confusion Matrix  [rows=actual, cols=predicted]")
    print(f"              Pred:Fake  Pred:Real")
    print(f"  Actual:Fake   {cm[0,0]:>6}     {cm[0,1]:>6}   (TN / FP)")
    print(f"  Actual:Real   {cm[1,0]:>6}     {cm[1,1]:>6}   (FN / TP)")

    print_section("Classification Report")
    print(classification_report(y_true, preds, target_names=["fake", "real"],
                                 digits=4))

    # Find the best possible threshold
    best_t, best_acer = find_best_threshold(y_true, y_scores)
    if abs(best_t - threshold) > 0.01:
        ba, bb, _ = compute_acer(y_true, y_scores, best_t)
        print_section("Optimal Threshold (for reference)")
        print(f"  Best threshold : {best_t:.4f}")
        print(f"  Best ACER      : {best_acer:.4f}")
        print(f"  APCER / BPCER  : {ba:.4f} / {bb:.4f}")

    return {
        "threshold": threshold,
        "apcer": apcer, "bpcer": bpcer, "acer": acer,
        "auc": auc, "eer": eer,
        "tp": int(cm[1,1]), "tn": int(cm[0,0]),
        "fp": int(cm[0,1]), "fn": int(cm[1,0]),
        "n_samples": len(y_true),
    }


def per_subject_report(y_true, y_scores, subjects, threshold):
    print_section("Per-Subject Breakdown")
    print(f"  {'Subject':>10} | {'N':>5} | {'Real':>5} | {'Fake':>5} | "
          f"{'APCER':>6} | {'BPCER':>6} | {'ACER':>6}")
    print(f"  {'-'*10}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-"
          f"{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    per_subj = {}
    for s in sorted(set(subjects)):
        m = subjects == s
        yt, ys = y_true[m], y_scores[m]
        if len(yt) == 0:
            continue
        n_real = (yt == 1).sum()
        n_fake = (yt == 0).sum()
        apcer, bpcer, acer = compute_acer(yt, ys, threshold)
        print(f"  {s:>10} | {len(yt):>5} | {n_real:>5} | {n_fake:>5} | "
              f"{apcer:>6.3f} | {bpcer:>6.3f} | {acer:>6.3f}")
        per_subj[s] = {"n": len(yt), "apcer": apcer, "bpcer": bpcer, "acer": acer}

    return per_subj


# ── Single-image mode ──────────────────────────────────────────────────────────

def predict_single(model, rgb_path, fft_path, device, threshold):
    rgb = VAL_TRANSFORMS(Image.open(rgb_path).convert("RGB")).unsqueeze(0).to(device)
    fft = load_fft(fft_path).unsqueeze(0).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(rgb, fft)).item()
    label = "REAL" if score >= threshold else "FAKE"
    confidence = score if label == "REAL" else 1 - score
    print(f"\n  Image   : {rgb_path}")
    print(f"  FFT     : {fft_path}")
    print(f"  Score   : {score:.4f}  (threshold: {threshold:.4f})")
    print(f"  Result  : {label}  (confidence: {confidence*100:.1f}%)")
    return score, label


# ── Main ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Anti-spoofing model evaluation")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pth checkpoint")
    p.add_argument("--csv", default=None,
                   help="Path to data.csv for batch evaluation")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="Only evaluate these subject IDs (e.g. 0003 0008)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override threshold (default: use saved threshold or 0.5)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-preds", default=None,
                   help="Save per-sample predictions to this CSV path")
    p.add_argument("--save-json", default=None,
                   help="Save summary metrics to this JSON path")
    # Single-image mode
    p.add_argument("--single-rgb", default=None,
                   help="Path to a single RGB image for inference")
    p.add_argument("--single-fft", default=None,
                   help="Path to a single FFT .npy file for inference")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice     : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_threshold = ckpt.get("threshold", 0.5)
    saved_epoch     = ckpt.get("epoch", "?")
    saved_acer      = ckpt.get("acer", "?")
    saved_auc       = ckpt.get("auc", "?")

    print(f"Saved epoch: {saved_epoch}  |  ACER={saved_acer}  |  AUC={saved_auc}")
    print(f"Saved threshold: {saved_threshold:.4f}")

    model = AntiSpoofNet(dropout=0.5).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")

    # Determine threshold
    threshold = args.threshold if args.threshold is not None else saved_threshold
    print(f"Using threshold: {threshold:.4f}")

    # ── Single-image mode ────────────────────────────────────────────────────
    if args.single_rgb is not None:
        if args.single_fft is None:
            print("ERROR: --single-fft is required with --single-rgb", file=sys.stderr)
            sys.exit(1)
        predict_single(model, args.single_rgb, args.single_fft, device, threshold)
        return

    # ── Batch evaluation mode ─────────────────────────────────────────────────
    if args.csv is None:
        print("ERROR: provide --csv for batch evaluation or --single-rgb for single image",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    # Normalize path separators (Windows backslashes -> forward slashes)
    df["rgb_path"] = df["rgb_path"].str.replace("\\", "/")
    df["fft_path"] = df["fft_path"].str.replace("\\", "/")
    print(f"\nTotal rows in CSV : {len(df)}")

    # Filter to existing files
    valid = df["rgb_path"].apply(os.path.exists) & df["fft_path"].apply(os.path.exists)
    df = df[valid].reset_index(drop=True)
    print(f"Valid file pairs  : {len(df)}")

    if len(df) == 0:
        print("ERROR: no valid samples found.", file=sys.stderr)
        sys.exit(1)

    # Add subject ID column
    df["subject_id"] = df["rgb_path"].apply(extract_subject_id)

    # Filter by subjects if requested
    if args.subjects:
        df = df[df["subject_id"].isin(args.subjects)].reset_index(drop=True)
        print(f"After subject filter: {len(df)} samples  "
              f"(subjects: {sorted(df['subject_id'].unique())})")
        if len(df) == 0:
            print("ERROR: no samples match the given subject IDs.", file=sys.stderr)
            sys.exit(1)

    # Build loader
    dataset = EvalDataset(df)

    def collate(batch):
        rgb, fft, lbl, sub, pth = zip(*batch)
        return (torch.stack(rgb), torch.stack(fft),
                torch.tensor(lbl), list(sub), list(pth))

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device == "cuda"), collate_fn=collate)

    # ── Run inference ─────────────────────────────────────────────────────────
    y_true, y_scores, subjects, paths = run_inference(model, loader, device)

    # ── Reports ───────────────────────────────────────────────────────────────
    summary = full_report(y_true, y_scores, threshold, args)
    per_subj = per_subject_report(y_true, y_scores, subjects, threshold)

    # ── Check for suspiciously good results (possible leakage) ─────────────
    if summary["auc"] > 0.99 and summary["acer"] < 0.01:
        print("\n  ⚠  WARNING: near-perfect results — possible identity leakage.")
        print("     Check that val subjects were NOT seen during training.\n")

    # ── Save predictions CSV ──────────────────────────────────────────────────
    if args.save_preds:
        preds_df = pd.DataFrame({
            "rgb_path":   paths,
            "subject_id": subjects,
            "label":      y_true.astype(int),
            "score":      y_scores,
            "predicted":  (y_scores >= threshold).astype(int),
            "correct":    (y_true == (y_scores >= threshold).astype(int)).astype(int),
        })
        preds_df.to_csv(args.save_preds, index=False)
        print(f"\n  Predictions saved to: {args.save_preds}")

    # ── Save summary JSON ─────────────────────────────────────────────────────
    if args.save_json:
        out = {**summary, "per_subject": per_subj,
               "checkpoint": args.checkpoint,
               "subjects_evaluated": sorted(set(subjects.tolist()))}
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Summary JSON saved to: {args.save_json}")

    print()


if __name__ == "__main__":
    main()
