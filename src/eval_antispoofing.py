import os, re, numpy as np, pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH   = "checkpoints/best_model_antispoofing.pth"
CSV_PATH     = "./dataset/processed_dataset/data.csv"
VAL_SUBJECTS = {"0001", "0003", "0008", "0015"}
IMG_SIZE     = 224
BATCH_SIZE   = 64
DEVICE       = torch.device("mps" if torch.backends.mps.is_available() else
                            "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Model (must match train_antispoofing.py exactly) ──────────────────────────
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

    def forward(self, x):
        return self.net(x).flatten(1)


class AntiSpoofNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.rgb_branch = base.features
        self.rgb_pool   = nn.AdaptiveAvgPool2d(1)
        self.fft_branch = FFTBranch()
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 2304, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, rgb, fft):
        r = self.rgb_pool(self.rgb_branch(rgb)).flatten(1)
        f = self.fft_branch(fft)
        return self.classifier(torch.cat([r, f], dim=1)).squeeze(1)


# ── Dataset ───────────────────────────────────────────────────────────────────
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_fft(path):
    arr = np.load(path).astype(np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return torch.from_numpy(arr).unsqueeze(0)

class ValDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb = VAL_TRANSFORMS(Image.open(row["rgb_path"]).convert("RGB"))
        fft = load_fft(row["fft_path"])
        return rgb, fft, torch.tensor(float(row["label"]), dtype=torch.float32)


# ── Build val split (same logic as training) ─────────────────────────────────
def extract_subject_id(filepath):
    stem = Path(filepath).stem
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", stem)
    if m:
        return m.group(1)
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem

df = pd.read_csv(CSV_PATH)
df = df[df["rgb_path"].apply(os.path.exists) &
        df["fft_path"].apply(os.path.exists)].reset_index(drop=True)
df["subject_id"] = df["rgb_path"].apply(extract_subject_id)

nuaa_mask = df["subject_id"].str.match(r"^\d{4}$")
# NUAA val subjects + 20% random sample of non-NUAA
nuaa_val     = df[nuaa_mask & df["subject_id"].isin(VAL_SUBJECTS)]
non_nuaa_val = df[~nuaa_mask].sample(frac=0.2, random_state=42)
val_df = pd.concat([nuaa_val, non_nuaa_val]).reset_index(drop=True)

print(f"Val set: {len(val_df)} samples  "
      f"(real={int((val_df['label']==1).sum())}, fake={int((val_df['label']==0).sum())})")

loader = DataLoader(ValDataset(val_df), batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=0)

# ── Load model ────────────────────────────────────────────────────────────────
model = AntiSpoofNet().to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
threshold = ckpt.get("threshold", 0.5)
print(f"Loaded model  |  saved threshold: {threshold:.4f}")

# ── Inference ─────────────────────────────────────────────────────────────────
all_scores, all_labels = [], []
with torch.no_grad():
    for rgb, fft, labels in loader:
        rgb, fft = rgb.to(DEVICE), fft.to(DEVICE)
        scores = torch.sigmoid(model(rgb, fft)).cpu().numpy()
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.numpy().tolist())

y_true   = np.array(all_labels)
y_scores = np.array(all_scores)
y_pred   = (y_scores >= threshold).astype(int)
classes  = ["fake", "real"]

# ── Report ────────────────────────────────────────────────────────────────────
auc = roc_auc_score(y_true, y_scores)
print(f"\n{'='*50}")
print("CLASSIFICATION REPORT")
print(f"{'='*50}")
print(classification_report(y_true, y_pred, target_names=classes, digits=4))
print(f"AUC-ROC: {auc:.4f}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
    ax=ax, colorbar=True, cmap="Blues")
ax.set_title("Anti-Spoofing — Confusion Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig("anti_spoofing_confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved: anti_spoofing_confusion_matrix.png")
