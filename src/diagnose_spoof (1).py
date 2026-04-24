"""
diagnose_spoof.py
=================
Run this and hold your real face in front of the camera for ~30 seconds.
It prints every raw score and a summary at the end.
This tells you exactly where to set the threshold — or whether the model
needs retraining because it can't distinguish your webcam at all.

Usage:
    python diagnose_spoof.py
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from collections import deque

SPOOF_MODEL_PATH = "./checkpoints/best_model_antispoofing.pth"
CAMERA_SOURCE    = 0
CROP_MARGIN      = 0.20


# ── Model (same as app.py) ────────────────────────────────────────────────────

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


SPOOF_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def fft_tensor(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (80, 80))
    mag  = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)
    mag  = (mag - mag.mean()) / (mag.std() + 1e-8)
    return torch.from_numpy(mag.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def crop_face(frame_bgr, bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    H, W = frame_bgr.shape[:2]
    mx = int(w * CROP_MARGIN); my = int(h * CROP_MARGIN)
    cx1 = max(0, x1 - mx); cy1 = max(0, y1 - my)
    cx2 = min(W, x2 + mx); cy2 = min(H, y2 + my)
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    return crop if crop.size > 0 else None


# ── Load model ────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

ckpt = torch.load(SPOOF_MODEL_PATH, map_location=device, weights_only=False)
saved_threshold = float(ckpt.get("threshold", 0.5))
model = AntiSpoofNet().to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model loaded. Saved threshold = {saved_threshold:.4f}\n")


# ── Face detector ─────────────────────────────────────────────────────────────

try:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(min_face_size=80, thresholds=[0.6, 0.7, 0.7],
                  keep_all=False, post_process=False, device=device)
    use_mtcnn = True
    print("Using MTCNN for face detection.")
except ImportError:
    use_mtcnn = False
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("MTCNN not found — falling back to Haar cascade.")

print("\n" + "="*60)
print("  Hold your REAL FACE in front of the camera.")
print("  Then hold up a FAKE (phone/printed photo).")
print("  Press Q to quit and see the summary.")
print("="*60 + "\n")


# ── Main loop ─────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(CAMERA_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

all_scores  = []
score_buf   = deque(maxlen=12)
frame_count = 0

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display = frame.copy()
        bbox = None

        # ── Detect face ───────────────────────────────────────────────────────
        if use_mtcnn:
            boxes, probs = mtcnn.detect(rgb)
            if boxes is not None and probs[0] > 0.85:
                bbox = boxes[0].astype(int)
        else:
            gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces  = face_cascade.detectMultiScale(gray_f, 1.1, 5, minSize=(80,80))
            if len(faces):
                x, y, w, h = faces[0]
                bbox = [x, y, x+w, y+h]

        if bbox is not None:
            crop = crop_face(frame, bbox)
            if crop is not None:
                # ── Inference ─────────────────────────────────────────────────
                rgb_t = SPOOF_TRANSFORM(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                ).unsqueeze(0).to(device)
                fft_t = fft_tensor(crop).to(device)

                raw      = torch.sigmoid(model(rgb_t, fft_t)).item()
                score_buf.append(raw)
                smoothed = sum(score_buf) / len(score_buf)
                all_scores.append(raw)

                is_real  = smoothed >= saved_threshold
                color    = (0, 200, 0) if is_real else (0, 0, 220)
                label    = f"{'REAL' if is_real else 'SPOOF'}  raw={raw:.3f}  smooth={smoothed:.3f}"

                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                cv2.putText(display, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Print every 15 frames so terminal doesn't flood
                if frame_count % 15 == 0:
                    verdict = "REAL  ✅" if is_real else "SPOOF ❌"
                    print(f"  frame {frame_count:>5} | raw={raw:.4f}  smooth={smoothed:.4f}  → {verdict}")

        # ── HUD ───────────────────────────────────────────────────────────────
        h_img, w_img = display.shape[:2]
        bar_y = h_img - 30
        cv2.rectangle(display, (0, bar_y), (w_img, h_img), (20,20,20), -1)

        if all_scores:
            fill = int(w_img * min(max(score_buf[-1] if score_buf else 0, 0), 1))
            s_fill = int(w_img * min(max(smoothed if score_buf else 0, 0), 1))
            t_x = int(w_img * saved_threshold)
            cv2.rectangle(display, (0, bar_y+2),  (fill,   bar_y+9),  (70,70,70), -1)
            cv2.rectangle(display, (0, bar_y+10), (s_fill, bar_y+18), color,       -1)
            cv2.line(display, (t_x, bar_y+2), (t_x, bar_y+18), (255,255,255), 2)
            txt = f"raw={all_scores[-1]:.3f}  smooth={smoothed:.3f}  threshold={saved_threshold:.3f}"
        else:
            txt = "No face detected"

        cv2.putText(display, txt, (8, h_img-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200,200,200), 1)
        cv2.putText(display, "Q = quit", (w_img-100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
        cv2.imshow("Spoof Diagnostic", display)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

cap.release()
cv2.destroyAllWindows()


# ── Summary ───────────────────────────────────────────────────────────────────

if not all_scores:
    print("\nNo scores recorded — no face was detected.")
else:
    arr  = np.array(all_scores)
    mean = arr.mean()
    std  = arr.std()
    mn   = arr.min()
    mx   = arr.max()
    pct_above = (arr >= saved_threshold).mean() * 100

    print("\n" + "="*60)
    print("  DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"  Frames scored     : {len(arr)}")
    print(f"  Mean score        : {mean:.4f}")
    print(f"  Std deviation     : {std:.4f}")
    print(f"  Min               : {mn:.4f}")
    print(f"  Max               : {mx:.4f}")
    print(f"  Saved threshold   : {saved_threshold:.4f}")
    print(f"  % above threshold : {pct_above:.1f}%  (→ would be called REAL)")
    print()

    if pct_above < 30:
        suggested = max(0.05, mean - 0.05)
        print(f"  ❌ MODEL LIKELY BROKEN for your webcam domain.")
        print(f"     Your real face scores around {mean:.3f} — way below threshold.")
        print(f"     Option A: lower threshold to ~{suggested:.3f} in menu option 5.")
        print(f"     Option B (better): retrain with webcam real + fake samples.")
    elif pct_above < 70:
        suggested = max(0.05, mean - 0.08)
        print(f"  ⚠️  Borderline — threshold is too high for your camera.")
        print(f"     Try lowering threshold to ~{suggested:.3f} in menu option 5.")
    else:
        print(f"  ✅ Model looks okay. Threshold {saved_threshold:.4f} seems reasonable.")
        print(f"     If still failing, try raising threshold slightly.")

    print()
    print("  Score distribution (histogram):")
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(arr, bins=bins)
    for i, count in enumerate(hist):
        lo, hi  = bins[i], bins[i+1]
        bar     = "█" * int(count / max(hist) * 30) if max(hist) > 0 else ""
        marker  = " ← threshold" if lo <= saved_threshold < hi else ""
        print(f"    {lo:.1f}-{hi:.1f}  {bar} {count}{marker}")
    print()
