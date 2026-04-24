"""
Face Recognition Attendance System — with Anti-Spoofing
=========================================================
FIXES APPLIED:
  1. SPOOF_TRANSFORM now resizes to 224x224 (was 80x80) — MobileNetV2 fix
  2. CLAHE removed from _prepare_spoof_crop (training had no CLAHE)
  3. Buffer sizes reduced: 24→12, confirm frames 8→4
  4. FFT computed from 80x80 grayscale (unchanged, was already correct)
"""

import csv
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import json, os, time
from collections import deque
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "camera_source":      0,
    "enrollment_frames":  20,

    "min_face_px":        80,
    "sharpness_min":      60.0,
    "brightness_min":     40,
    "brightness_max":     230,

    "embedding_buffer_size": 12,
    "label_stable_frames":    8,
    "recognition_threshold":  0.60,

    "spoof_model_path":          "./checkpoints/best_model_antispoofing.pth",
    "spoof_buffer_size":          16,   
    "spoof_real_confirm_frames":   8,   
    "spoof_confirm_frames":        6,   
    "spoof_crop_margin":           0.30,

    "gallery_path": "./face_gallery.json",
}


# ─────────────────────────────────────────────────────────────
# ANTI-SPOOFING MODEL
# ─────────────────────────────────────────────────────────────

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


# FIX 1: resize to 224x224 to match MobileNetV2 training size (was 80x80)
SPOOF_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _prepare_spoof_crop(frame_bgr, bbox):
    """
    Crop face with extra margin.
    FIX 2: CLAHE removed — training data had no CLAHE, applying it at
    inference shifts the pixel distribution and confuses the model.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    H, W = frame_bgr.shape[:2]

    m  = CONFIG["spoof_crop_margin"]
    mx = int(w * m);  my = int(h * m)
    cx1 = max(0, x1 - mx);  cy1 = max(0, y1 - my)
    cx2 = min(W, x2 + mx);  cy2 = min(H, y2 + my)
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    return crop


def _fft_tensor(face_bgr):
    """FFT computed from 80x80 grayscale — matches training."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (80, 80))
    mag  = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)
    mag  = (mag - mag.mean()) / (mag.std() + 1e-8)
    return torch.from_numpy(mag.astype(np.float32)).unsqueeze(0).unsqueeze(0)


class AntiSpoofPredictor:
    def __init__(self, device):
        self.device = device
        ckpt = torch.load(CONFIG["spoof_model_path"],
                          map_location=device, weights_only=False)
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.model = AntiSpoofNet().to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"✅ Anti-spoof model loaded  (threshold={self.threshold:.4f})")

        self.score_buffer  = deque(maxlen=CONFIG["spoof_buffer_size"])
        self.REAL_CONFIRM  = CONFIG["spoof_real_confirm_frames"]
        self.SPOOF_CONFIRM = CONFIG["spoof_confirm_frames"]
        self._consec_real  = 0
        self._consec_spoof = 0
        self._stable_state = "real"
        self.all_scores    = []

    @torch.no_grad()
    def predict(self, face_bgr):
        """Returns (is_real, smoothed_score, warming_up)."""
        rgb = SPOOF_TRANSFORM(
            cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        ).unsqueeze(0).to(self.device)
        fft = _fft_tensor(face_bgr).to(self.device)

        raw = torch.sigmoid(self.model(rgb, fft)).item()
        self.score_buffer.append(raw)
        self.all_scores.append(raw)
        smoothed   = sum(self.score_buffer) / len(self.score_buffer)
        warming_up = len(self.score_buffer) < CONFIG["spoof_buffer_size"]

        if smoothed >= self.threshold:
            self._consec_real  += 1;  self._consec_spoof  = 0
        else:
            self._consec_spoof += 1;  self._consec_real   = 0

        if self._consec_real  >= self.REAL_CONFIRM:
            self._stable_state = "real"
        elif self._consec_spoof >= self.SPOOF_CONFIRM:
            self._stable_state = "spoof"

        return self._stable_state == "real", smoothed, warming_up

    def reset(self):
        self.score_buffer.clear()
        self._consec_real = self._consec_spoof = 0
        self._stable_state = "real"

    def score_summary(self):
        if not self.all_scores:
            print("  No scores recorded yet.")
            return
        arr  = self.all_scores
        mean = sum(arr) / len(arr)
        print(f"\n── Spoof score summary ({len(arr)} frames) ──────────────")
        print(f"  Mean : {mean:.4f}   Min : {min(arr):.4f}   Max : {max(arr):.4f}")
        print(f"  Saved threshold : {self.threshold:.4f}")
        pct = sum(1 for s in arr if s >= self.threshold) / len(arr) * 100
        print(f"  Frames above threshold (→ REAL) : {pct:.1f}%")
        if pct < 60:
            sug = max(0.05, mean - 0.05)
            print(f"  ⚠️  Your real face scores below threshold too often.")
            print(f"     Suggested threshold : {sug:.4f}  (use menu option 5)")
        elif pct > 95:
            print(f"  ✅ Threshold looks well-calibrated for your camera.")
        else:
            sug = mean - 0.05
            print(f"  ⚠️  Borderline — consider lowering threshold to ~{sug:.4f}")


# ─────────────────────────────────────────────────────────────
# FACE DETECTOR  (MTCNN)
# ─────────────────────────────────────────────────────────────
class FaceDetector:
    def __init__(self, device):
        from facenet_pytorch import MTCNN
        self.embed_mtcnn = MTCNN(
            image_size=112, margin=15,
            min_face_size=CONFIG["min_face_px"],
            thresholds=[0.6, 0.7, 0.7],
            keep_all=False, post_process=True, device=device,
        )
        self.display_mtcnn = MTCNN(
            min_face_size=CONFIG["min_face_px"],
            thresholds=[0.6, 0.7, 0.7],
            keep_all=False, post_process=False, device=device,
        )

    def detect(self, frame_bgr):
        """Returns (face_tensor, bbox, spoof_crop) or (None, None, None)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face_tensor, prob = self.embed_mtcnn(rgb, return_prob=True)
        if face_tensor is None or prob < 0.90:
            return None, None, None

        boxes, _ = self.display_mtcnn.detect(rgb)
        if boxes is None:
            return face_tensor, None, None

        bbox       = boxes[0].astype(int)
        spoof_crop = _prepare_spoof_crop(frame_bgr, bbox)
        return face_tensor, bbox, spoof_crop


# ─────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────
class EmbeddingModel:
    def __init__(self, device):
        from mobilefacenet import MobileFaceNet
        self.device = device
        print("⏳ Loading custom MobileFaceNet model ...")
        self.model = MobileFaceNet(embedding_size=512).to(device)
        ckpt = torch.load("./checkpoints/best_model.pth", map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print("✅ Identity model ready")

    @torch.no_grad()
    def embed(self, face_tensor):
        x = face_tensor.unsqueeze(0).to(self.device)
        return F.normalize(self.model(x), p=2, dim=1).squeeze(0).cpu()


# ─────────────────────────────────────────────────────────────
# QUALITY GATE
# ─────────────────────────────────────────────────────────────
def frame_quality(frame_bgr, bbox):
    if bbox is None:
        return False, "No face"
    x1, y1, x2, y2 = bbox
    w, h = x2-x1, y2-y1
    if w < CONFIG["min_face_px"] or h < CONFIG["min_face_px"]:
        return False, f"Move closer ({w}x{h} px)"
    crop = frame_bgr[max(0,y1):y2, max(0,x1):x2]
    if crop.size == 0:
        return False, "Bad crop"
    gray       = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness  = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    if sharpness  < CONFIG["sharpness_min"]:  return False, f"Blurry ({sharpness:.0f})"
    if brightness < CONFIG["brightness_min"]: return False, f"Too dark ({brightness:.0f})"
    if brightness > CONFIG["brightness_max"]: return False, f"Too bright ({brightness:.0f})"
    return True, "Good"


# ─────────────────────────────────────────────────────────────
# FACE GALLERY
# ─────────────────────────────────────────────────────────────
class FaceGallery:
    def __init__(self):
        self.path    = CONFIG["gallery_path"]
        self.gallery = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                raw = json.load(f)
            self.gallery = {k: torch.tensor(v) for k, v in raw.items()}
            print(f"✅ Gallery: {list(self.gallery.keys())}")
        else:
            print("ℹ️  Empty gallery")

    def _save(self):
        with open(self.path, "w") as f:
            json.dump({k: v.tolist() for k, v in self.gallery.items()}, f)

    def enroll(self, name, embeddings):
        mean_emb = F.normalize(torch.stack(embeddings).mean(dim=0), p=2, dim=0)
        self.gallery[name] = mean_emb
        self._save()
        print(f"✅ '{name}' enrolled ({len(embeddings)} frames)")

    def recognize(self, query_emb):
        if not self.gallery:
            return "Unknown", 0.0
        scores = {n: F.cosine_similarity(query_emb.unsqueeze(0),
                                          e.unsqueeze(0)).item()
                  for n, e in self.gallery.items()}
        best, score = max(scores.items(), key=lambda x: x[1])
        return (best, score) if score >= CONFIG["recognition_threshold"] \
               else ("Unknown", score)

    def delete(self, name):
        if name in self.gallery:
            del self.gallery[name]; self._save()

    def names(self):
        return list(self.gallery.keys())


# ─────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────
def open_camera():
    cap = cv2.VideoCapture(CONFIG["camera_source"])
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,      1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 Camera: {w}x{h}")
    return cap


# ─────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────
def draw_box_label(img, bbox, label, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    lbl_y = y1-32 if y1 > 38 else y2+6
    cv2.rectangle(img, (x1,lbl_y), (x1+tw+8, lbl_y+th+8), color, -1)
    cv2.putText(img, label, (x1+4, lbl_y+th+2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

def draw_score_bar(img, bbox, score, color):
    x1, y1, x2, y2 = bbox
    fill = int((x2-x1) * min(max(score,0.0),1.0))
    cv2.rectangle(img, (x1,y2+4), (x2,     y2+12), (50,50,50), -1)
    cv2.rectangle(img, (x1,y2+4), (x1+fill,y2+12), color,      -1)

def draw_spoof_hud(img, raw, smoothed, state, warming_up, threshold):
    h, w = img.shape[:2]
    bar_y = h - 30
    cv2.rectangle(img, (0,bar_y), (w,h), (20,20,20), -1)

    fill_raw  = int(w * min(max(raw,      0), 1))
    fill_smoo = int(w * min(max(smoothed, 0), 1))
    t_x       = int(w * threshold)

    cv2.rectangle(img, (0,bar_y+2),  (fill_raw,  bar_y+9),  (70,70,70),   -1)
    color = (0,200,255) if warming_up else \
            (0,220,0)   if state == "real" else (0,0,220)
    cv2.rectangle(img, (0,bar_y+10), (fill_smoo, bar_y+18), color,         -1)
    cv2.line(img, (t_x,bar_y+2), (t_x,bar_y+18), (255,255,255), 2)

    if warming_up:
        txt = f"Warming up... raw={raw:.3f}"
    else:
        txt = f"raw={raw:.3f}  smooth={smoothed:.3f}  thr={threshold:.3f}  → {state.upper()}"
    cv2.putText(img, txt, (8, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

def draw_attendance_overlay(img, attendance):
    ox = img.shape[1] - 230
    cv2.putText(img, "Attendance", (ox,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    for i, (person, t) in enumerate(attendance.items()):
        cv2.putText(img, f"  {person}  {t}", (ox, 44+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,210,0), 1)


# ─────────────────────────────────────────────────────────────
# ENROLLMENT
# ─────────────────────────────────────────────────────────────
PROMPTS = [
    "Look straight",      "Turn slightly LEFT",   "Turn slightly RIGHT",
    "Tilt head UP",       "Tilt head DOWN",        "Smile",
    "Neutral expression", "Move CLOSER",           "Move FURTHER BACK",
    "Look straight again",
]

def enroll_person(name, detector, embedder, gallery):
    cap = open_camera()
    if cap is None:
        return False

    target = CONFIG["enrollment_frames"]
    collected, prompt_idx, prompt_tick = [], 0, time.time()
    print(f"\n📸 Enrolling '{name}' — need {target} good frames.")
    print(f"   Press ESC or close the window to cancel\n")

    window_name = f"Enrolling: {name}"

    try:
        while len(collected) < target:
            ret, frame = cap.read()
            if not ret:
                break

            face_tensor, bbox, _ = detector.detect(frame)
            ok, msg = frame_quality(frame, bbox)
            display = frame.copy()

            n = len(collected)
            cv2.rectangle(display, (10,10), (310,32), (40,40,40), -1)
            cv2.rectangle(display, (10,10), (10+int(300*n/target),32), (0,210,0), -1)
            cv2.putText(display, f"{name}  {n}/{target}", (10,55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if time.time() - prompt_tick > 2.5:
                prompt_idx  = (prompt_idx+1) % len(PROMPTS)
                prompt_tick = time.time()
            cv2.putText(display, PROMPTS[prompt_idx], (10,82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,230,230), 2)

            if bbox is not None:
                x1,y1,x2,y2 = bbox
                cv2.rectangle(display,(x1,y1),(x2,y2),
                              (0,210,0) if ok else (0,60,220),2)
            cv2.putText(display, msg, (10,display.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0,210,0) if ok else (0,80,220),2)
            cv2.imshow(window_name, display)

            if ok and face_tensor is not None:
                collected.append(embedder.embed(face_tensor))
                time.sleep(0.12)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print(f"⏸️  Enrollment cancelled.")
                return False
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print(f"⏸️  Enrollment cancelled.")
                return False
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(collected) >= 8:
        gallery.enroll(name, collected)
        return True
    print(f"❌ Only {len(collected)} frames.")
    return False


# ─────────────────────────────────────────────────────────────
# RECOGNITION  with anti-spoofing
# ─────────────────────────────────────────────────────────────
def run_recognition(detector, embedder, gallery, spoof_predictor):
    cap = open_camera()
    if cap is None:
        return

    emb_buffer    = deque(maxlen=CONFIG["embedding_buffer_size"])
    stable_name   = "..."
    stable_score  = 0.0
    candidate     = None
    candidate_run = 0

    spoof_predictor.reset()
    last_raw     = 0.0
    last_smoothed= 0.0
    last_state   = "real"
    last_warming = True

    attendance = {}
    log_path = "./attendance_log.csv"
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["name", "date", "time"])

    print("\n🎯 Recognition + Anti-Spoofing — Press ESC to quit")
    print("   Watch the bottom bar to see your live spoof scores.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_tensor, bbox, spoof_crop = detector.detect(frame)
            display = frame.copy()

            if face_tensor is None or spoof_crop is None:
                emb_buffer.clear()
                stable_name = "..."; stable_score = 0.0
                candidate = None;    candidate_run = 0
                spoof_predictor.reset()
                last_warming = True

            else:
                is_real, smoothed, warming_up = spoof_predictor.predict(spoof_crop)
                last_raw      = spoof_predictor.score_buffer[-1] \
                                if spoof_predictor.score_buffer else 0.0
                last_smoothed = smoothed
                last_state    = "real" if is_real else "spoof"
                last_warming  = warming_up

                if warming_up:
                    if bbox is not None:
                        buf     = len(spoof_predictor.score_buffer)
                        buf_max = CONFIG["spoof_buffer_size"]
                        pct     = int((bbox[2]-bbox[0]) * buf / buf_max)
                        cv2.rectangle(display,(bbox[0],bbox[1]-8),
                                      (bbox[2],bbox[1]),(40,40,40),-1)
                        cv2.rectangle(display,(bbox[0],bbox[1]-8),
                                      (bbox[0]+pct,bbox[1]),(0,180,255),-1)
                        cv2.rectangle(display,(bbox[0],bbox[1]),
                                      (bbox[2],bbox[3]),(0,180,255),2)
                        cv2.putText(display,f"Checking {buf}/{buf_max}",
                                    (bbox[0],bbox[1]-12),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,180,255),1)

                elif not is_real:
                    emb_buffer.clear()
                    stable_name = "..."; candidate = None; candidate_run = 0
                    if bbox is not None:
                        draw_box_label(display, bbox,
                                       f"SPOOF  {smoothed:.2f}", (0,0,255))
                        draw_score_bar(display, bbox, smoothed, (0,0,255))

                else:
                    emb_buffer.append(embedder.embed(face_tensor))
                    avg_emb = F.normalize(
                        torch.stack(list(emb_buffer)).mean(dim=0), p=2, dim=0)
                    raw_name, raw_score = gallery.recognize(avg_emb)

                    if raw_name == candidate:
                        candidate_run += 1
                    else:
                        candidate = raw_name; candidate_run = 1

                    if candidate_run >= CONFIG["label_stable_frames"]:
                        stable_name  = candidate
                        stable_score = raw_score

                    if bbox is not None:
                        known     = stable_name not in ("Unknown", "...")
                        box_color = (0,200,0) if known else (130,130,130)
                        label     = f"LIVE {smoothed:.2f} | {stable_name} {stable_score:.2f}"
                        draw_box_label(display, bbox, label, box_color)
                        draw_score_bar(display, bbox, stable_score, box_color)

                        if known and stable_name not in attendance:
                            now = datetime.now()
                            d = now.strftime("%Y-%m-%d")
                            t = now.strftime("%H:%M:%S")
                            attendance[stable_name] = f"{d} {t}"
                            with open(log_path, "a", newline="") as f:
                                csv.writer(f).writerow([stable_name, d, t])
                            print(f"✅ Attendance: {stable_name} {d} {t}")

            draw_spoof_hud(display, last_raw, last_smoothed,
                           last_state, last_warming, spoof_predictor.threshold)
            draw_attendance_overlay(display, attendance)
            cv2.imshow("Attendance  |  ESC = quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    spoof_predictor.score_summary()
    print(f"\n✅ Saved → {log_path}")
    for person, t in attendance.items():
        print(f"   {person}  {t}")


# ─────────────────────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'█'*56}")
    print("  🎭 Face Attendance + Anti-Spoofing")
    print(f"  ⚙️  Device: {device}")
    print(f"{'█'*56}\n")

    detector        = FaceDetector(device)
    embedder        = EmbeddingModel(device)
    gallery         = FaceGallery()
    spoof_predictor = AntiSpoofPredictor(device)

    while True:
        enrolled = gallery.names()
        print(f"\n┌─ Status ─────────────────────────────────────────┐")
        if enrolled:
            print(f"│ 👥 Enrolled      : {', '.join(enrolled):<32} │")
        else:
            print(f"│ 👥 Enrolled      : (none)                         │")
        print(f"│ 🛡️  Live Threshold : {spoof_predictor.threshold:<8.4f}                    │")
        print(f"└──────────────────────────────────────────────────┘")

        print(f"\n┌─ Menu ────────────────────────────────────────────┐")
        print(f"│                                                  │")
        print(f"│  1  🎞️   Enroll new person                       │")
        print(f"│  2  ▶️   Start attendance (camera + detection)   │")
        print(f"│  3  🗑️   Delete person                           │")
        print(f"│  4  ⚖️   Adjust recognition threshold           │")
        print(f"│  5  🎚️   Adjust liveness threshold              │")
        print(f"│  6  🚪  Quit                                    │")
        print(f"│                                                  │")
        print(f"└──────────────────────────────────────────────────┘")

        choice = input("\n  Choose option (1-6): ").strip()

        if choice == "1":
            name = input("\n  Name for enrollment: ").strip()
            if name:
                enroll_person(name, detector, embedder, gallery)
                print("  ✅ Done!")
            else:
                print("  ⚠️  Cancelled.")

        elif choice == "2":
            if not gallery.gallery:
                print("\n  ⚠️  No one enrolled yet. Please enroll at least one person.")
            else:
                run_recognition(detector, embedder, gallery, spoof_predictor)

        elif choice == "3":
            if not gallery.gallery:
                print("\n  ⚠️  Gallery is empty.")
            else:
                print(f"\n  Available: {', '.join(gallery.names())}")
                name = input("  Delete who? → ").strip()
                if name in gallery.names():
                    gallery.delete(name)
                    print(f"  ✅ Deleted '{name}'")
                else:
                    print(f"  ⚠️  '{name}' not found.")

        elif choice == "4":
            print(f"\n  📊 Current recognition threshold: {CONFIG['recognition_threshold']:.4f}")
            print(f"  (Higher = stricter, must be 0.5-1.0)")
            try:
                new_val = float(input("  New threshold: "))
                if 0.5 <= new_val <= 1.0:
                    CONFIG["recognition_threshold"] = new_val
                    print(f"  ✅ Set to {new_val:.4f}")
                else:
                    print(f"  ⚠️  Must be between 0.5 and 1.0")
            except ValueError:
                print("  ⚠️  Invalid input.")

        elif choice == "5":
            print(f"\n📊 Current liveness threshold : {spoof_predictor.threshold:.4f}")
            spoof_predictor.score_summary()
            print(f"\n  Lower → easier to pass as real   |   Raise → stricter")
            try:
                new_t = float(input("\n  New threshold: "))
                spoof_predictor.threshold = new_t
                spoof_predictor.reset()
                print(f"  ✅ Set to {new_t:.4f}")
            except ValueError:
                print("  ⚠️  Invalid input.")

        elif choice == "6":
            print("\n  👋 Goodbye!\n")
            break

        else:
            print("  ⚠️  Invalid option. Please choose 1-6.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  👋 Stopped by user. Goodbye!\n")