"""
NUAA Dataset Preprocessor for Face Anti-Spoofing
=================================================
NUAA folder structure expected:
    NUAA/
        ClientFace/       <- real faces (subfolders 0001/, 0002/, ...)
        ImposterFace/     <- fake/spoof faces (subfolders 0001/, 0002/, ...)

Output:
    processed_dataset/
        real/   <- cropped + aligned real face images
        fake/   <- cropped + aligned spoof face images
        data.csv <- path, label (1=real, 0=fake)
"""

import os
import cv2
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
NUAA_ROOT      = "./NUAA"               # change to your NUAA root path
OUTPUT_DIR     = "./processed_dataset"
IMG_SIZE       = 80                     # resize face crop to 80x80
MARGIN         = 0.3                    # margin around detected face (30%)
MIN_FACE_SIZE  = 40                     # skip faces smaller than this (px)

REAL_DIRS = ["ClientRaw"]
FAKE_DIRS = ["ImposterRaw"]
# ──────────────────────────────────────────────────────────────────────────────

def get_face_cascade():
    """Use OpenCV Haar cascade as lightweight face detector."""
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

    # Take largest detected face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # Add margin
    mx = int(w * margin)
    my = int(h * margin)
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)

    face_crop = img_bgr[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (img_size, img_size))
    return face_resized

def compute_fft_map(img_bgr, size=80):
    """
    Compute log-magnitude Fourier spectrum of grayscale face.
    Used as a second input channel to detect screen/print artifacts.
    Returns (size x size) float32 array normalized to [0, 1].
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (size, size))
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1e-8)
    # Normalize to [0, 1]
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max - mag_min > 0:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    return magnitude.astype(np.float32)

def process_directory(src_dir, label, output_subdir, cascade, records):
    """
    Walk src_dir recursively, detect face in each image,
    save cropped face + FFT map to output_subdir.
    Appends (rgb_path, fft_path, label) to records list.
    """
    os.makedirs(output_subdir, exist_ok=True)
    fft_dir = output_subdir + "_fft"
    os.makedirs(fft_dir, exist_ok=True)

    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        img_paths.extend(Path(src_dir).rglob(ext))

    skipped = 0
    for img_path in tqdm(img_paths, desc=f"  {'real' if label==1 else 'fake'}"):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        face = detect_and_crop(img, cascade, IMG_SIZE, MARGIN)
        if face is None:
            skipped += 1
            continue

        # Build output filename (preserve subfolder structure)
        rel = img_path.relative_to(src_dir)
        flat_name = str(rel).replace(os.sep, "_")
        stem = Path(flat_name).stem

        rgb_out  = os.path.join(output_subdir, stem + ".jpg")
        fft_out  = os.path.join(fft_dir,       stem + ".npy")

        cv2.imwrite(rgb_out, face)

        fft_map = compute_fft_map(face, IMG_SIZE)
        np.save(fft_out, fft_map)

        records.append((rgb_out, fft_out, label))

    print(f"    Skipped (no face / unreadable): {skipped}")
    return records

def run():
    cascade = get_face_cascade()
    records = []

    print("\n── Processing REAL faces ──────────────────────────────────────")
    for d in REAL_DIRS:
        src = os.path.join(NUAA_ROOT, d)
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found, skipping.")
            continue
        out = os.path.join(OUTPUT_DIR, "real")
        process_directory(src, label=1, output_subdir=out, cascade=cascade, records=records)

    print("\n── Processing FAKE faces ──────────────────────────────────────")
    for d in FAKE_DIRS:
        src = os.path.join(NUAA_ROOT, d)
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found, skipping.")
            continue
        out = os.path.join(OUTPUT_DIR, "fake")
        process_directory(src, label=0, output_subdir=out, cascade=cascade, records=records)

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rgb_path", "fft_path", "label"])
        writer.writerows(records)

    real_count = sum(1 for r in records if r[2] == 1)
    fake_count = sum(1 for r in records if r[2] == 0)
    print(f"\n── Done ───────────────────────────────────────────────────────")
    print(f"  Real samples : {real_count}")
    print(f"  Fake samples : {fake_count}")
    print(f"  Total        : {len(records)}")
    print(f"  CSV saved to : {csv_path}")

if __name__ == "__main__":
    run()
