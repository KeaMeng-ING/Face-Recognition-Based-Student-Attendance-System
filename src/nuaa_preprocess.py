"""
NUAA Dataset Preprocessor for Face Anti-Spoofing
=================================================
Processes multiple attack types into a single combined dataset:

  Source 1 — NUAA (printed photo attacks):
    NUAA/ClientRaw/      <- real faces
    NUAA/ImposterRaw/    <- printed photo spoofs

  Source 2 — Personal fake data (screen / video attacks):
    my_spoof_data/fake/  <- phone screen, phone video, etc.

Output:
    processed_dataset/
        real/        <- cropped real face images
        fake/        <- cropped fake face images (all attack types)
        real_fft/    <- FFT maps for real faces
        fake_fft/    <- FFT maps for fake faces
        data.csv     <- rgb_path, fft_path, label, attack_type
"""

import os
import cv2
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
NUAA_ROOT    = "./dataset/NUAA"
EXTRA_FAKE   = "./my_spoof_data/fake"   # personal phone screen / video attacks
OUTPUT_DIR   = "./dataset/processed_dataset"
IMG_SIZE     = 80
MARGIN       = 0.3
MIN_FACE_SIZE = 40

REAL_DIRS = ["ClientRaw"]
FAKE_DIRS = ["ImposterRaw"]
# ──────────────────────────────────────────────────────────────────────────────


def get_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def detect_and_crop(img_bgr, cascade, img_size=80, margin=0.3):
    """
    Detect largest face in image, add margin, resize to img_size x img_size.
    Returns cropped face (numpy array) or None if no face found.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )
    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    mx = int(w * margin)
    my = int(h * margin)
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, x - mx);  y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx);  y2 = min(h_img, y + h + my)

    face_crop = img_bgr[y1:y2, x1:x2]
    return cv2.resize(face_crop, (img_size, img_size))


def compute_fft_map(img_bgr, size=80):
    """
    Log-magnitude Fourier spectrum — detects screen/print texture artifacts.
    Returns (size x size) float32 array normalised to [0, 1].
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (size, size))
    magnitude = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max - mag_min > 0:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    return magnitude.astype(np.float32)


def process_directory(src_dir, label, output_subdir, cascade, records, attack_type):
    """
    Walk src_dir recursively, detect face in each image,
    save cropped face + FFT map to output_subdir.
    Appends (rgb_path, fft_path, label, attack_type) to records.
    """
    os.makedirs(output_subdir, exist_ok=True)
    fft_dir = output_subdir + "_fft"
    os.makedirs(fft_dir, exist_ok=True)

    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        img_paths.extend(Path(src_dir).rglob(ext))

    skipped = 0
    for img_path in tqdm(img_paths, desc=f"  {attack_type}"):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        face = detect_and_crop(img, cascade, IMG_SIZE, MARGIN)
        if face is None:
            skipped += 1
            continue

        # Flat filename to avoid subfolder collisions — prefix with attack type
        rel       = img_path.relative_to(src_dir)
        flat_name = str(rel).replace(os.sep, "_")
        stem      = f"{attack_type}_{Path(flat_name).stem}"

        rgb_out = os.path.join(output_subdir, stem + ".jpg")
        fft_out = os.path.join(fft_dir,       stem + ".npy")

        cv2.imwrite(rgb_out, face)
        np.save(fft_out, compute_fft_map(face, IMG_SIZE))

        records.append((rgb_out, fft_out, label, attack_type))

    print(f"    Skipped (no face / unreadable): {skipped}")
    return records


def run():
    cascade = get_face_cascade()
    records = []

    # ── Real faces (NUAA ClientRaw) ───────────────────────────────────────────
    print("\n── Processing REAL faces (NUAA) ───────────────────────────────")
    for d in REAL_DIRS:
        src = os.path.join(NUAA_ROOT, d)
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found, skipping.")
            continue
        out = os.path.join(OUTPUT_DIR, "real")
        process_directory(src, label=1, output_subdir=out,
                          cascade=cascade, records=records,
                          attack_type="real_nuaa")

    # ── Fake: printed photos (NUAA ImposterRaw) ───────────────────────────────
    print("\n── Processing FAKE faces — printed photo (NUAA) ───────────────")
    for d in FAKE_DIRS:
        src = os.path.join(NUAA_ROOT, d)
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found, skipping.")
            continue
        out = os.path.join(OUTPUT_DIR, "fake")
        process_directory(src, label=0, output_subdir=out,
                          cascade=cascade, records=records,
                          attack_type="printed_photo")

    # ── Fake: personal screen / video attacks ─────────────────────────────────
    if os.path.exists(EXTRA_FAKE):
        # Each subfolder is a different attack type (e.g. shared/, phone_screen/, etc.)
        subdirs = [d for d in Path(EXTRA_FAKE).iterdir() if d.is_dir()]
        if subdirs:
            print("\n── Processing FAKE faces — personal attacks ───────────────────")
            for subdir in sorted(subdirs):
                # Infer attack type from filenames inside (phone_screen, phone_video, etc.)
                sample_files = list(subdir.glob("*.jpg"))[:1]
                if sample_files:
                    attack_type = "_".join(sample_files[0].stem.split("_")[:-1]) or subdir.name
                else:
                    attack_type = subdir.name

                # If mixed types exist in same folder, split by filename prefix
                all_files = list(subdir.glob("*.jpg"))
                prefixes = set("_".join(f.stem.split("_")[:-1]) for f in all_files)

                if len(prefixes) > 1:
                    # Multiple attack types in one folder — process each prefix separately
                    for prefix in sorted(prefixes):
                        grouped = [f for f in all_files
                                   if "_".join(f.stem.split("_")[:-1]) == prefix]
                        print(f"\n  Found attack type: {prefix} ({len(grouped)} images)")
                        out = os.path.join(OUTPUT_DIR, "fake")
                        os.makedirs(out, exist_ok=True)
                        fft_dir = out + "_fft"
                        os.makedirs(fft_dir, exist_ok=True)

                        skipped = 0
                        for img_path in tqdm(grouped, desc=f"  {prefix}"):
                            img = cv2.imread(str(img_path))
                            if img is None:
                                skipped += 1; continue
                            face = detect_and_crop(img, cascade, IMG_SIZE, MARGIN)
                            if face is None:
                                skipped += 1; continue

                            stem    = f"{prefix}_{img_path.stem.split('_')[-1]}"
                            rgb_out = os.path.join(out,     stem + ".jpg")
                            fft_out = os.path.join(fft_dir, stem + ".npy")
                            cv2.imwrite(rgb_out, face)
                            np.save(fft_out, compute_fft_map(face, IMG_SIZE))
                            records.append((rgb_out, fft_out, 0, prefix))

                        print(f"    Skipped (no face / unreadable): {skipped}")
                else:
                    out = os.path.join(OUTPUT_DIR, "fake")
                    process_directory(str(subdir), label=0, output_subdir=out,
                                      cascade=cascade, records=records,
                                      attack_type=attack_type)
        else:
            print(f"\n  No subfolders found in {EXTRA_FAKE}, skipping personal fakes.")
    else:
        print(f"\n  {EXTRA_FAKE} not found — skipping personal attack types.")
        print("  (Only NUAA printed photo attacks will be in the dataset)")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rgb_path", "fft_path", "label", "attack_type"])
        writer.writerows(records)

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    attack_counts = Counter(r[3] for r in records)

    real_count = sum(1 for r in records if r[2] == 1)
    fake_count = sum(1 for r in records if r[2] == 0)

    print(f"\n── Done ────────────────────────────────────────────────────────")
    print(f"  Real samples    : {real_count}")
    print(f"  Fake samples    : {fake_count}")
    print(f"  Total           : {len(records)}")
    print(f"\n  Breakdown by attack type:")
    for attack, count in sorted(attack_counts.items()):
        label = "real" if "real" in attack else "fake"
        print(f"    {attack:<20}  {count:>5}  ({label})")
    print(f"\n  CSV saved to    : {csv_path}")


if __name__ == "__main__":
    run()
