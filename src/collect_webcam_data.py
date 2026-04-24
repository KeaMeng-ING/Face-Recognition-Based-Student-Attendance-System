"""
collect_webcam_data.py
======================
Collects real face + fake (phone/printed photo) samples from your webcam.
Saves cropped face images + FFT .npy files ready for retraining.

Usage:
    python collect_webcam_data.py --label real --n 300
    python collect_webcam_data.py --label fake --n 300

Tips for FAKE samples:
  - Show your face photo on a phone screen
  - Show a printed photo
  - Try different angles, distances, lighting
"""

import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

try:
    from facenet_pytorch import MTCNN
    USE_MTCNN = True
except ImportError:
    USE_MTCNN = False

SAVE_ROOT    = "./dataset/webcam_data"
CROP_MARGIN  = 0.20
IMG_SIZE     = 224   # saved RGB size (matches MobileNetV2 training)
FFT_SIZE     = 80    # FFT computed at 80x80 (matches FFTBranch training)
CAMERA_SOURCE = 0


def get_face_bbox(frame_bgr, mtcnn=None, cascade=None):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if mtcnn is not None:
        boxes, probs = mtcnn.detect(rgb)
        if boxes is not None and probs[0] > 0.85:
            return boxes[0].astype(int)
    elif cascade is not None:
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces):
            x, y, w, h = faces[0]
            return np.array([x, y, x+w, y+h])
    return None


def crop_with_margin(frame_bgr, bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    H, W = frame_bgr.shape[:2]
    mx = int(w * CROP_MARGIN); my = int(h * CROP_MARGIN)
    cx1 = max(0, x1 - mx); cy1 = max(0, y1 - my)
    cx2 = min(W, x2 + mx); cy2 = min(H, y2 + my)
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    return crop if crop.size > 0 else None


def save_sample(crop_bgr, out_dir, index):
    """Save RGB image + FFT npy."""
    rgb_path = str(out_dir / f"{index:05d}.jpg")
    fft_path = str(out_dir / f"{index:05d}_fft.npy")

    # RGB — resize to 224x224
    resized = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(rgb_path, resized)

    # FFT — 80x80 grayscale, log magnitude, normalized
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (FFT_SIZE, FFT_SIZE))
    mag  = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)
    mag  = (mag - mag.mean()) / (mag.std() + 1e-8)
    np.save(fft_path, mag.astype(np.float32))

    return rgb_path, fft_path


def collect(label: str, n_target: int):
    assert label in ("real", "fake"), "label must be 'real' or 'fake'"

    out_dir = Path(SAVE_ROOT) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count existing samples so we don't overwrite
    existing = len(list(out_dir.glob("*.jpg")))
    print(f"\nSaving to: {out_dir}")
    print(f"Existing samples: {existing}")
    print(f"Target new samples: {n_target}")

    device = torch.device("cpu")
    mtcnn = cascade = None
    if USE_MTCNN:
        mtcnn = MTCNN(min_face_size=80, thresholds=[0.6, 0.7, 0.7],
                      keep_all=False, post_process=False, device=device)
        print("Using MTCNN")
    else:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("Using Haar cascade")

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    collected = 0
    frame_idx = 0
    # Save every N frames to get diversity (not burst of identical frames)
    SAVE_EVERY = 3

    instructions = {
        "real": [
            "Look straight at camera",
            "Turn head slightly LEFT",
            "Turn head slightly RIGHT",
            "Tilt head UP",
            "Tilt head DOWN",
            "Move CLOSER to camera",
            "Move FURTHER from camera",
            "Different lighting if possible",
        ],
        "fake": [
            "Show PHONE SCREEN with your photo",
            "Tilt phone slightly",
            "Move phone closer",
            "Move phone further",
            "Show PRINTED PHOTO if available",
            "Try different angles",
        ],
    }
    hint_list = instructions[label]
    hint_idx  = 0
    last_hint_time = 0

    COLOR = (0, 200, 0) if label == "real" else (0, 0, 220)
    LABEL_TEXT = "REAL FACE" if label == "real" else "FAKE / SPOOF"

    print(f"\n{'='*55}")
    print(f"  Collecting: {LABEL_TEXT}")
    print(f"  Need {n_target} samples. Press Q to stop early.")
    print(f"{'='*55}\n")

    while collected < n_target:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        bbox = get_face_bbox(frame, mtcnn=mtcnn, cascade=cascade)
        display = frame.copy()
        face_found = False

        import time
        now = time.time()
        if now - last_hint_time > 4.0:
            hint_idx = (hint_idx + 1) % len(hint_list)
            last_hint_time = now

        if bbox is not None:
            crop = crop_with_margin(frame, bbox)
            if crop is not None:
                face_found = True

                if frame_idx % SAVE_EVERY == 0:
                    idx = existing + collected
                    save_sample(crop, out_dir, idx)
                    collected += 1
                    if collected % 30 == 0:
                        print(f"  Collected {collected}/{n_target}")

                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), COLOR, 2)

        # Progress bar
        pct = collected / n_target
        bar_w = int(display.shape[1] * pct)
        cv2.rectangle(display, (0, 0), (display.shape[1], 18), (30, 30, 30), -1)
        cv2.rectangle(display, (0, 0), (bar_w, 18), COLOR, -1)
        cv2.putText(display, f"{LABEL_TEXT}  {collected}/{n_target}",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2)
        cv2.putText(display, hint_list[hint_idx],
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
        status = "Face detected" if face_found else "No face — move into frame"
        cv2.putText(display, status,
                    (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 0) if face_found else (80, 80, 220), 2)

        cv2.imshow(f"Collecting: {LABEL_TEXT}", display)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            print("  Stopped early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    total = existing + collected
    print(f"\n✅ Done. Collected {collected} new samples. Total in '{label}': {total}")
    return collected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=["real", "fake"],
                        help="'real' for your live face, 'fake' for spoof attacks")
    parser.add_argument("--n", type=int, default=300,
                        help="Number of samples to collect (default: 300)")
    args = parser.parse_args()
    collect(args.label, args.n)
