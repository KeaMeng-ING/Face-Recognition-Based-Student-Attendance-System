"""
ANTI-SPOOFING MODEL - COMPREHENSIVE EVALUATION
═══════════════════════════════════════════════════════════════════════

Generates:
  ✓ Confusion Matrix (REAL vs FAKE)
  ✓ ROC Curve
  ✓ Score Distribution
  ✓ Overall Accuracy, Precision, Recall, F1
  ✓ ACER, APCER, BPCER metrics
  ✓ Classification Report
  ✓ Text Report File
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    roc_auc_score
)


# ══════════════════════════════════════════════════════════════════════
# ANTI-SPOOFING MODEL ARCHITECTURE 
# ══════════════════════════════════════════════════════════════════════

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
            nn.AdaptiveAvgPool2d((3, 3)),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class AntiSpoofNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.rgb_branch = base.features
        self.rgb_pool = nn.AdaptiveAvgPool2d(1)
        self.fft_branch = FFTBranch()
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 2304, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, rgb, fft):
        r = self.rgb_pool(self.rgb_branch(rgb)).flatten(1)
        f = self.fft_branch(fft)
        return self.classifier(torch.cat([r, f], dim=1)).squeeze(1)


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_fft(path: str) -> torch.Tensor:
    """Load FFT with normalization"""
    arr = np.load(path).astype(np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return torch.from_numpy(arr).unsqueeze(0)


VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_acer(y_true, y_scores, threshold=0.5):
    """Compute APCER, BPCER, ACER"""
    preds = (y_scores >= threshold).astype(int)
    fake_mask = y_true == 0
    real_mask = y_true == 1
    apcer = (preds[fake_mask] == 1).mean() if fake_mask.any() else 0.0
    bpcer = (preds[real_mask] == 0).mean() if real_mask.any() else 0.0
    return float(apcer), float(bpcer), float((apcer + bpcer) / 2)


# ══════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_antispoofing(checkpoint_path, csv_path, output_dir='./checkpoints', threshold=None):
    """
    Complete evaluation of anti-spoofing model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # ── Load checkpoint ────────────────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_threshold = ckpt.get('threshold', 0.5)
    saved_epoch = ckpt.get('epoch', '?')
    saved_acer = ckpt.get('acer', '?')
    saved_auc = ckpt.get('auc', '?')

    if threshold is None:
        threshold = saved_threshold

    print(f"  Checkpoint: epoch={saved_epoch}, ACER={saved_acer}, AUC={saved_auc}")
    print(f"  Using threshold: {threshold:.4f}\n")

    # ── Load model ─────────────────────────────────────────────────────
    model = AntiSpoofNet(dropout=0.5).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # ── Load dataset ───────────────────────────────────────────────────
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df["rgb_path"] = df["rgb_path"].str.replace("\\", "/")
    df["fft_path"] = df["fft_path"].str.replace("\\", "/")

    # Fix paths - prepend dataset/ to make them correct from demo root
    df["rgb_path"] = "dataset/" + df["rgb_path"]
    df["fft_path"] = "dataset/" + df["fft_path"]

    # Filter valid files
    valid = df["rgb_path"].apply(os.path.exists) & df["fft_path"].apply(os.path.exists)
    df = df[valid].reset_index(drop=True)
    print(f"Total samples: {len(df)}\n")

    # ── Inference ──────────────────────────────────────────────────────
    print("Running inference...")
    all_labels = []
    all_scores = []
    errors = []

    for idx in tqdm(range(len(df)), desc="Predicting"):
        row = df.iloc[idx]
        try:
            rgb = VAL_TRANSFORMS(Image.open(row["rgb_path"]).convert("RGB")).unsqueeze(0).to(device)
            fft = load_fft(row["fft_path"]).unsqueeze(0).to(device)

            with torch.no_grad():
                logit = model(rgb, fft)
                score = torch.sigmoid(logit).item()

            all_scores.append(score)
            all_labels.append(int(row["label"]))
        except Exception as e:
            errors.append((row["rgb_path"], str(e)))
            continue

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    y_pred = (y_scores >= threshold).astype(int)

    if errors:
        print(f"\n⚠️  {len(errors)} errors during inference (skipped)")

    # ── Compute metrics ────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("ANTI-SPOOFING MODEL EVALUATION")
    print("═" * 70)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_true, y_scores)

    apcer, bpcer, acer = compute_acer(y_true, y_scores, threshold)

    cm = confusion_matrix(y_true, y_pred)

    print("\n📊 OVERALL METRICS:")
    print("-" * 70)
    print(f"  Accuracy:  {accuracy*100:6.2f}%")
    print(f"  Precision: {precision*100:6.2f}%")
    print(f"  Recall:    {recall*100:6.2f}%")
    print(f"  F1-Score:  {f1*100:6.2f}%")
    print(f"  AUC-ROC:   {auc_score:6.4f}")

    print("\n📋 CONFUSION MATRIX:")
    print("-" * 70)
    print(f"                Predicted")
    print(f"          REAL      FAKE")
    print(f"  REAL  {cm[1,1]:5d}    {cm[1,0]:5d}  (TN / FN)")
    print(f"  FAKE  {cm[0,1]:5d}    {cm[0,0]:5d}  (FP / TP)")

    print("\n🛡️  LIVENESS METRICS:")
    print("-" * 70)
    print(f"  APCER (Fake→Real error):  {apcer*100:6.2f}%")
    print(f"  BPCER (Real→Fake error):  {bpcer*100:6.2f}%")
    print(f"  ACER (Average):           {acer*100:6.2f}%")
    print(f"  Threshold used:           {threshold:6.4f}")

    print("\n📈 SAMPLE STATS:")
    print("-" * 70)
    print(f"  Real faces:  {(y_true == 1).sum()}")
    print(f"  Fake faces:  {(y_true == 0).sum()}")

    # ── Create visualizations ──────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_data = np.array([[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]])
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE', 'REAL'],
                yticklabels=['REAL', 'FAKE'],
                cbar_kws={'label': 'Count'},
                ax=ax)
    ax.set_title('Anti-Spoofing Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    cm_path = f'{output_dir}/antispoofing_confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {cm_path}")
    plt.close()

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'AUC = {auc_score:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC=0.5)')
    ax.scatter([fpr[np.argmin(np.abs(thresholds - threshold))]],
               [tpr[np.argmin(np.abs(thresholds - threshold))]],
               marker='o', s=200, color='red', label=f'Threshold={threshold:.3f}', zorder=5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Anti-Spoofing Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = f'{output_dir}/antispoofing_roc_curve.png'
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {roc_path}")
    plt.close()

    # 3. Score Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_scores[y_true == 1], bins=50, alpha=0.6, label='Real Faces', color='green', edgecolor='black')
    ax.hist(y_scores[y_true == 0], bins=50, alpha=0.6, label='Fake Faces', color='red', edgecolor='black')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2.5, label=f'Threshold={threshold:.3f}')
    ax.set_xlabel('Model Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    dist_path = f'{output_dir}/antispoofing_score_distribution.png'
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {dist_path}")
    plt.close()

    # 4. Text Report
    report_txt = f"""{'='*70}
ANTI-SPOOFING MODEL - COMPREHENSIVE EVALUATION REPORT
{'='*70}

CHECKPOINT INFO:
  Path:      {checkpoint_path}
  Epoch:     {saved_epoch}
  Saved ACER: {saved_acer}
  Saved AUC:  {saved_auc}

EVALUATION DATA:
  CSV Path:  {csv_path}
  Total Samples: {len(df)}
  Real Faces: {(y_true == 1).sum()}
  Fake Faces: {(y_true == 0).sum()}

{'='*70}
OVERALL METRICS
{'='*70}
  Accuracy:   {accuracy*100:.2f}%
  Precision:  {precision*100:.2f}%
  Recall:     {recall*100:.2f}%
  F1-Score:   {f1*100:.2f}%
  AUC-ROC:    {auc_score:.4f}

{'='*70}
LIVENESS-SPECIFIC METRICS
{'='*70}
  APCER (Attack→Real):  {apcer*100:.2f}%
    ↳ % of fake faces incorrectly classified as real
  BPCER (Real→Fake):    {bpcer*100:.2f}%
    ↳ % of real faces incorrectly classified as fake
  ACER (Average):       {acer*100:.2f}%
    ↳ (APCER + BPCER) / 2

  Threshold Used:       {threshold:.4f}

{'='*70}
CONFUSION MATRIX
{'='*70}
              Predicted
          REAL      FAKE
  REAL  {cm[1,1]:5d}    {cm[1,0]:5d}  (TN / FN)
  FAKE  {cm[0,1]:5d}    {cm[0,0]:5d}  (FP / TP)

  True Negatives (REAL→REAL):   {cm[1,1]}
  False Negatives (REAL→FAKE):  {cm[1,0]}
  False Positives (FAKE→REAL):  {cm[0,1]}
  True Positives (FAKE→FAKE):   {cm[0,0]}

{'='*70}
CLASSIFICATION REPORT
{'='*70}
"""
    class_report = classification_report(y_true, y_pred,
                                        labels=[0, 1],
                                        target_names=['Fake', 'Real'],
                                        digits=4)
    report_txt += class_report

    report_path = f'{output_dir}/antispoofing_evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_txt)
    print(f"✓ Saved: {report_path}")

    print("\n" + "═" * 70)
    print("✅ EVALUATION COMPLETE")
    print("═" * 70)
    print("\nGenerated files:")
    print(f"  📊 {cm_path}")
    print(f"  📈 {roc_path}")
    print(f"  📉 {dist_path}")
    print(f"  📄 {report_path}")


if __name__ == "__main__":
    evaluate_antispoofing(
        checkpoint_path='./checkpoints/best_model_antispoofing.pth',
        csv_path='./dataset/processed_dataset/data.csv',
        output_dir='./checkpoints',
        threshold=None  # Use saved threshold from checkpoint
    )
