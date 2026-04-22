"""
Dataset Explorer
================
Prints statistics and saves visualisation plots for all three datasets
used in this project:

  1. VGGFace2      — face identity / recognition dataset
  2. NUAA          — anti-spoofing dataset (raw + processed)
  3. Personal data — your own collected real/spoof samples

Usage:
    python src/understand_dataset.py

Outputs (saved to ./checkpoints/dataset_analysis/):
    vggface2_samples.png         — sample faces from VGGFace2
    nuaa_samples.png             — side-by-side real vs fake from NUAA
    personal_samples.png         — your own real vs fake samples
    class_distribution.png      — bar chart of all dataset sizes
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
VGGFACE2_DIR  = "./dataset/vggface2_train"
NUAA_REAL_RAW = "./dataset/NUAA/ClientRaw"
NUAA_FAKE_RAW = "./dataset/NUAA/ImposterRaw"
NUAA_REAL_PRO = "./dataset/processed_dataset/real"
NUAA_FAKE_PRO = "./dataset/processed_dataset/fake"
PERSONAL_REAL = "./my_spoof_data/real"
PERSONAL_FAKE = "./my_spoof_data/fake"
OUT_DIR       = "./checkpoints/dataset_analysis"
# ──────────────────────────────────────────────────────────────────────────────


def count_images(folder, exts=("*.jpg", "*.jpeg", "*.png")):
    """Count all image files recursively in a folder."""
    total = 0
    for ext in exts:
        total += len(list(Path(folder).rglob(ext)))
    return total


def sample_images(folder, n=8, seed=42):
    """Return n random image paths from a folder (recursive)."""
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(Path(folder).rglob(ext))
    if not files:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
    return [str(files[i]) for i in sorted(idx)]


def load_rgb(path, size=(112, 112)):
    """Load an image as RGB numpy array, resized."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((*size, 3), dtype=np.uint8)
    img = cv2.resize(img, size)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_stats(paths):
    """Compute mean brightness and sharpness for a list of image paths."""
    brightness_list, sharpness_list = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_list.append(float(gray.mean()))
        sharpness_list.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    if not brightness_list:
        return 0, 0
    return np.mean(brightness_list), np.mean(sharpness_list)


# ── 1. Print statistics ───────────────────────────────────────────────────────

def print_vggface2_stats():
    print("\n" + "═" * 60)
    print("  DATASET 1 — VGGFace2 (Face Recognition)")
    print("═" * 60)

    identities = [d for d in Path(VGGFACE2_DIR).iterdir() if d.is_dir()]
    img_counts = [count_images(str(d)) for d in identities]

    print(f"  Identities        : {len(identities)}")
    print(f"  Total images      : {sum(img_counts)}")
    print(f"  Avg per identity  : {np.mean(img_counts):.1f}")
    print(f"  Min / Max         : {min(img_counts)} / {max(img_counts)}")

    # Top 5 largest identities
    top = sorted(zip(identities, img_counts), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Top 5 identities by image count:")
    for d, n in top:
        print(f"    {d.name:<15}  {n} images")

    return identities, img_counts


def print_nuaa_stats():
    print("\n" + "═" * 60)
    print("  DATASET 2 — NUAA Anti-Spoofing")
    print("═" * 60)

    # Raw
    real_subjects = [d for d in Path(NUAA_REAL_RAW).iterdir() if d.is_dir()]
    fake_subjects = [d for d in Path(NUAA_FAKE_RAW).iterdir() if d.is_dir()]
    real_raw = count_images(NUAA_REAL_RAW)
    fake_raw = count_images(NUAA_FAKE_RAW)

    print(f"\n  RAW (before preprocessing):")
    print(f"    Real subjects   : {len(real_subjects)}")
    print(f"    Fake subjects   : {len(fake_subjects)}")
    print(f"    Real images     : {real_raw}")
    print(f"    Fake images     : {fake_raw}")
    print(f"    Total           : {real_raw + fake_raw}")
    print(f"    Real/Fake ratio : {real_raw/fake_raw:.2f}")

    # Processed
    real_pro = count_images(NUAA_REAL_PRO)
    fake_pro = count_images(NUAA_FAKE_PRO)
    dropped_real = real_raw - real_pro
    dropped_fake = fake_raw - fake_pro

    print(f"\n  PROCESSED (after face detection + crop):")
    print(f"    Real images     : {real_pro}  (dropped {dropped_real} — no face found)")
    print(f"    Fake images     : {fake_pro}  (dropped {dropped_fake} — no face found)")
    print(f"    Total           : {real_pro + fake_pro}")

    # Per-subject breakdown
    print(f"\n  Per-subject image counts (raw):")
    print(f"    {'Subject':<10}  {'Real':>6}  {'Fake':>6}")
    print(f"    {'-'*26}")
    for subj_dir in sorted(real_subjects, key=lambda d: d.name):
        subj = subj_dir.name
        r = count_images(str(subj_dir))
        fake_dir = Path(NUAA_FAKE_RAW) / subj
        f = count_images(str(fake_dir)) if fake_dir.exists() else 0
        print(f"    {subj:<10}  {r:>6}  {f:>6}")

    return real_pro, fake_pro


def print_personal_stats():
    print("\n" + "═" * 60)
    print("  DATASET 3 — Personal Data (Fine-tuning)")
    print("═" * 60)

    real_total = 0
    fake_total = 0

    real_folders = [d for d in Path(PERSONAL_REAL).iterdir() if d.is_dir()] \
                   if Path(PERSONAL_REAL).exists() else []
    fake_folders = [d for d in Path(PERSONAL_FAKE).iterdir() if d.is_dir()] \
                   if Path(PERSONAL_FAKE).exists() else []

    print(f"\n  Real samples (by person):")
    for d in sorted(real_folders):
        n = count_images(str(d))
        real_total += n
        bright, sharp = image_stats(sample_images(str(d), n=50))
        print(f"    {d.name:<25}  {n:>4} images  "
              f"brightness={bright:.1f}  sharpness={sharp:.1f}")

    print(f"\n  Fake samples (by type):")
    for d in sorted(fake_folders):
        n = count_images(str(d))
        fake_total += n
        bright, sharp = image_stats(sample_images(str(d), n=50))
        print(f"    {d.name:<25}  {n:>4} images  "
              f"brightness={bright:.1f}  sharpness={sharp:.1f}")

    print(f"\n  Total real  : {real_total}")
    print(f"  Total fake  : {fake_total}")
    print(f"  Total       : {real_total + fake_total}")
    print(f"  Balance     : {'balanced ✓' if abs(real_total - fake_total) < 50 else 'imbalanced ⚠'}")

    return real_total, fake_total


# ── 2. Save visualisations ────────────────────────────────────────────────────

def save_vggface2_samples():
    print("\n  Saving VGGFace2 sample grid...")
    identity_dirs = sorted(Path(VGGFACE2_DIR).iterdir())[:8]

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    fig.suptitle("VGGFace2 — Sample Faces (8 Identities × 2 Images)", fontsize=13)

    for col, id_dir in enumerate(identity_dirs):
        imgs = sample_images(str(id_dir), n=2)
        for row in range(2):
            ax = axes[row][col]
            if row < len(imgs):
                ax.imshow(load_rgb(imgs[row]))
            ax.axis("off")
            if row == 0:
                ax.set_title(id_dir.name, fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "vggface2_samples.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_nuaa_samples():
    print("  Saving NUAA sample grid...")
    real_imgs = sample_images(NUAA_REAL_PRO, n=8)
    fake_imgs = sample_images(NUAA_FAKE_PRO, n=8)

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    fig.suptitle("NUAA Anti-Spoofing — Real (top) vs Fake/Spoof (bottom)", fontsize=13)

    for col in range(8):
        # Real row
        ax = axes[0][col]
        if col < len(real_imgs):
            ax.imshow(load_rgb(real_imgs[col], size=(80, 80)))
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("REAL", fontsize=10, color="green", fontweight="bold")

        # Fake row
        ax = axes[1][col]
        if col < len(fake_imgs):
            ax.imshow(load_rgb(fake_imgs[col], size=(80, 80)))
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("FAKE", fontsize=10, color="red", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "nuaa_samples.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_personal_samples():
    print("  Saving personal data sample grid...")
    real_imgs = sample_images(PERSONAL_REAL, n=8)
    fake_imgs = sample_images(PERSONAL_FAKE, n=8)

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    fig.suptitle("Personal Data — Real (top) vs Fake/Spoof (bottom)", fontsize=13)

    for col in range(8):
        ax = axes[0][col]
        if col < len(real_imgs):
            ax.imshow(load_rgb(real_imgs[col]))
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("REAL", fontsize=10, color="green", fontweight="bold")

        ax = axes[1][col]
        if col < len(fake_imgs):
            ax.imshow(load_rgb(fake_imgs[col]))
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("FAKE", fontsize=10, color="red", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "personal_samples.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_class_distribution(vgg_total, nuaa_real, nuaa_fake, personal_real, personal_fake):
    print("  Saving class distribution chart...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Dataset Class Distribution", fontsize=14, fontweight="bold")

    # VGGFace2 — images per identity (histogram)
    identity_dirs = [d for d in Path(VGGFACE2_DIR).iterdir() if d.is_dir()]
    counts = [count_images(str(d)) for d in identity_dirs]
    axes[0].hist(counts, bins=20, color="steelblue", edgecolor="black")
    axes[0].set_title("VGGFace2\nImages per Identity")
    axes[0].set_xlabel("Images")
    axes[0].set_ylabel("Number of Identities")
    axes[0].grid(True, alpha=0.3)

    # NUAA — real vs fake
    axes[1].bar(["Real", "Fake"], [nuaa_real, nuaa_fake],
                color=["green", "red"], edgecolor="black", alpha=0.8)
    axes[1].set_title("NUAA Anti-Spoofing\n(Processed)")
    axes[1].set_ylabel("Image Count")
    for i, v in enumerate([nuaa_real, nuaa_fake]):
        axes[1].text(i, v + 30, str(v), ha="center", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Personal — real vs fake
    axes[2].bar(["Real", "Fake"], [personal_real, personal_fake],
                color=["green", "red"], edgecolor="black", alpha=0.8)
    axes[2].set_title("Personal Data\n(Fine-tuning)")
    axes[2].set_ylabel("Image Count")
    for i, v in enumerate([personal_real, personal_fake]):
        axes[2].text(i, v + 2, str(v), ha="center", fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "class_distribution.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Print stats
    identities, img_counts = print_vggface2_stats()
    nuaa_real, nuaa_fake   = print_nuaa_stats()
    personal_real, personal_fake = print_personal_stats()

    vgg_total = sum(img_counts)

    print("\n" + "═" * 60)
    print("  OVERALL SUMMARY")
    print("═" * 60)
    print(f"  VGGFace2   : {len(identities)} identities, {vgg_total:,} images")
    print(f"  NUAA       : {nuaa_real + nuaa_fake:,} images  (real={nuaa_real}, fake={nuaa_fake})")
    print(f"  Personal   : {personal_real + personal_fake} images  "
          f"(real={personal_real}, fake={personal_fake})")

    # Save plots
    print("\n── Saving visualisations ──────────────────────────────────")
    save_vggface2_samples()
    save_nuaa_samples()
    save_personal_samples()
    save_class_distribution(vgg_total, nuaa_real, nuaa_fake, personal_real, personal_fake)

    print(f"\n✅ All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
