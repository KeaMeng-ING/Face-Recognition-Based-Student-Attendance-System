"""
ITM-390 Final Project — Model Evaluation Script
Generates: classification reports, confusion matrices, training curves
Framework: PyTorch + facenet-pytorch
Run: python evaluate_models.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

# ─────────────────────────────────────────────
# CONFIG — edit these paths to match your setup
# ─────────────────────────────────────────────
FACE_RECOGNITION_MODEL_PATH = "face_recognition_model.pth"   # your saved .pth file
ANTI_SPOOFING_MODEL_PATH    = "anti_spoofing_model.pth"       # your saved .pth file

FACE_RECOGNITION_TEST_DIR   = "dataset/face_recognition/test"  # folder with one subfolder per student
ANTI_SPOOFING_TEST_DIR      = "dataset/anti_spoofing/test"      # folder with 'real/' and 'fake/' subfolders

FACE_RECOGNITION_HISTORY    = "face_recognition_history.json"  # training history JSON (optional)
ANTI_SPOOFING_HISTORY       = "anti_spoofing_history.json"      # training history JSON (optional)

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# IMAGE TRANSFORMS
# ─────────────────────────────────────────────
face_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

spoof_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# HELPER: save confusion matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2),
                                    max(5, len(class_names) * 1.0)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# HELPER: plot training curves from history dict
# ─────────────────────────────────────────────
def plot_training_curves(history: dict, title: str, save_path: str):
    """
    history keys expected: 'train_acc', 'val_acc', 'train_loss', 'val_loss'
    Each is a list of values, one per epoch.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", markersize=3, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   "r-o", markersize=3, label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# HELPER: run inference and collect predictions
# ─────────────────────────────────────────────
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ═════════════════════════════════════════════
# 1. FACE RECOGNITION EVALUATION
# ═════════════════════════════════════════════
print("\n" + "="*55)
print("1. FACE RECOGNITION MODEL EVALUATION")
print("="*55)

if not os.path.exists(FACE_RECOGNITION_TEST_DIR):
    print(f"  [SKIP] Test directory not found: {FACE_RECOGNITION_TEST_DIR}")
    print("  Edit FACE_RECOGNITION_TEST_DIR at the top of this script.")
else:
    # Load dataset
    fr_dataset = datasets.ImageFolder(FACE_RECOGNITION_TEST_DIR, transform=face_transform)
    fr_loader  = DataLoader(fr_dataset, batch_size=32, shuffle=False, num_workers=0)
    student_names = fr_dataset.classes
    num_students  = len(student_names)
    print(f"  Students found: {student_names}")

    # Load model — MobileFaceNet or your custom architecture
    # Adjust this block if you use a different model class
    try:
        from mobilefacenet import MobileFaceNet  # your local file
        fr_model = MobileFaceNet(num_classes=num_students)
    except ImportError:
        # Fallback: use MobileNetV2 backbone with custom head
        fr_model = models.mobilenet_v2(weights=None)
        fr_model.classifier[1] = nn.Linear(fr_model.last_channel, num_students)

    fr_model.load_state_dict(torch.load(FACE_RECOGNITION_MODEL_PATH, map_location=DEVICE))
    fr_model = fr_model.to(DEVICE)
    print("  Model loaded.")

    # Run evaluation
    y_true, y_pred = evaluate_model(fr_model, fr_loader, DEVICE)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=student_names, digits=4)
    print("\n  Classification Report (Face Recognition):")
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "face_recognition_report.txt")
    with open(report_path, "w") as f:
        f.write("Face Recognition — Classification Report\n")
        f.write("="*50 + "\n")
        f.write(report)
    print(f"  Saved: {report_path}")

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"  Overall Accuracy: {acc*100:.2f}%")

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, student_names,
        "Face Recognition — Confusion Matrix",
        os.path.join(OUTPUT_DIR, "face_recognition_confusion_matrix.png")
    )

# Training curves (if history file exists)
if os.path.exists(FACE_RECOGNITION_HISTORY):
    with open(FACE_RECOGNITION_HISTORY) as f:
        fr_history = json.load(f)
    plot_training_curves(
        fr_history,
        "Face Recognition — Training Curves",
        os.path.join(OUTPUT_DIR, "face_recognition_training_curves.png")
    )
else:
    print(f"\n  [INFO] No history file found at '{FACE_RECOGNITION_HISTORY}'.")
    print("  To generate training curves, save history during training (see below).")
    print("  Example at the bottom of this script shows how to do this.")


# ═════════════════════════════════════════════
# 2. ANTI-SPOOFING EVALUATION
# ═════════════════════════════════════════════
print("\n" + "="*55)
print("2. ANTI-SPOOFING MODEL EVALUATION")
print("="*55)

if not os.path.exists(ANTI_SPOOFING_TEST_DIR):
    print(f"  [SKIP] Test directory not found: {ANTI_SPOOFING_TEST_DIR}")
    print("  Edit ANTI_SPOOFING_TEST_DIR at the top of this script.")
else:
    # Load dataset — expects subfolders: real/, fake/
    as_dataset = datasets.ImageFolder(ANTI_SPOOFING_TEST_DIR, transform=spoof_transform)
    as_loader  = DataLoader(as_dataset, batch_size=64, shuffle=False, num_workers=0)
    spoof_classes = as_dataset.classes   # ['fake', 'real'] — alphabetical
    print(f"  Classes found: {spoof_classes}")

    # Load model — MobileNetV2 binary classifier
    as_model = models.mobilenet_v2(weights=None)
    as_model.classifier[1] = nn.Linear(as_model.last_channel, 2)
    as_model.load_state_dict(torch.load(ANTI_SPOOFING_MODEL_PATH, map_location=DEVICE))
    as_model = as_model.to(DEVICE)
    print("  Model loaded.")

    # Run evaluation
    y_true_as, y_pred_as = evaluate_model(as_model, as_loader, DEVICE)

    # Classification report
    report_as = classification_report(y_true_as, y_pred_as, target_names=spoof_classes, digits=4)
    print("\n  Classification Report (Anti-Spoofing):")
    print(report_as)

    report_as_path = os.path.join(OUTPUT_DIR, "anti_spoofing_report.txt")
    with open(report_as_path, "w") as f:
        f.write("Anti-Spoofing — Classification Report\n")
        f.write("="*50 + "\n")
        f.write(report_as)
    print(f"  Saved: {report_as_path}")

    acc_as = accuracy_score(y_true_as, y_pred_as)
    print(f"  Overall Accuracy: {acc_as*100:.2f}%")

    # Confusion matrix
    plot_confusion_matrix(
        y_true_as, y_pred_as, spoof_classes,
        "Anti-Spoofing — Confusion Matrix",
        os.path.join(OUTPUT_DIR, "anti_spoofing_confusion_matrix.png")
    )

# Training curves
if os.path.exists(ANTI_SPOOFING_HISTORY):
    with open(ANTI_SPOOFING_HISTORY) as f:
        as_history = json.load(f)
    plot_training_curves(
        as_history,
        "Anti-Spoofing — Training Curves",
        os.path.join(OUTPUT_DIR, "anti_spoofing_training_curves.png")
    )
else:
    print(f"\n  [INFO] No history file found at '{ANTI_SPOOFING_HISTORY}'.")

print("\nDone! All outputs saved to:", OUTPUT_DIR)


# ═════════════════════════════════════════════
# HOW TO SAVE TRAINING HISTORY DURING TRAINING
# ═════════════════════════════════════════════
"""
Add this to your train.py so you can generate training curves:

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(num_epochs):
    # --- your training loop ---
    train_loss = ...
    train_acc  = ...
    val_loss   = ...
    val_acc    = ...

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)    # as a float, e.g. 94.2
    history["val_acc"].append(val_acc)

# After training finishes, save it:
import json
with open("face_recognition_history.json", "w") as f:
    json.dump(history, f)
"""
