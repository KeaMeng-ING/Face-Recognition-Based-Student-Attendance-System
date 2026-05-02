"""
Streamlit Face Attendance System
Run from project root: streamlit run src/streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import json, time, csv, threading
from collections import deque
from datetime import datetime
import av
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

SPOOF_MODEL_PATH = os.path.join(ROOT_DIR, "checkpoints", "best_model_antispoofing.pth")
EMBED_MODEL_PATH = os.path.join(ROOT_DIR, "checkpoints", "best_model.pth")
GALLERY_PATH     = os.path.join(ROOT_DIR, "face_gallery.json")
LOG_PATH         = os.path.join(ROOT_DIR, "attendance_log.csv")

DEFAULT_CFG = {
    "min_face_px":               80,
    "sharpness_min":             60.0,
    "brightness_min":            40,
    "brightness_max":            230,
    "embedding_buffer_size":     12,
    "label_stable_frames":       8,
    "recognition_threshold":     0.60,
    "spoof_buffer_size":         5,
    "spoof_real_confirm_frames": 8,
    "spoof_confirm_frames":      6,
    "spoof_crop_margin":         0.30,
    "enrollment_frames":         20,
}

PROMPTS = [
    "Look straight",       "Turn slightly LEFT",  "Turn slightly RIGHT",
    "Tilt head UP",        "Tilt head DOWN",       "Smile",
    "Neutral expression",  "Move CLOSER",          "Move FURTHER BACK",
    "Look straight again",
]


# ── Model Definitions ──────────────────────────────────────────────────────────

class FFTBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
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
            nn.Linear(1280 + 2304, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
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


# ── Cached Loaders ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from facenet_pytorch import MTCNN
    embed_mtcnn = MTCNN(
        image_size=112, margin=15, min_face_size=DEFAULT_CFG["min_face_px"],
        thresholds=[0.6, 0.7, 0.7], keep_all=False, post_process=True, device=device,
    )
    display_mtcnn = MTCNN(
        min_face_size=DEFAULT_CFG["min_face_px"], thresholds=[0.6, 0.7, 0.7],
        keep_all=False, post_process=False, device=device,
    )

    from mobilefacenet import MobileFaceNet
    embedder = MobileFaceNet(embedding_size=512).to(device)
    ckpt = torch.load(EMBED_MODEL_PATH, map_location=device)
    embedder.load_state_dict(ckpt["model_state_dict"])
    embedder.eval()

    ckpt_spoof = torch.load(SPOOF_MODEL_PATH, map_location=device, weights_only=False)
    spoof_net  = AntiSpoofNet().to(device)
    spoof_net.load_state_dict(ckpt_spoof["model_state"])
    spoof_net.eval()
    spoof_default_thr = float(ckpt_spoof.get("threshold", 0.5))

    return device, embed_mtcnn, display_mtcnn, embedder, spoof_net, spoof_default_thr


@st.cache_resource
def get_gallery():
    data = {}
    if os.path.exists(GALLERY_PATH):
        with open(GALLERY_PATH) as f:
            raw = json.load(f)
        data = {k: torch.tensor(v) for k, v in raw.items()}
    return {"data": data, "lock": threading.Lock()}


def save_gallery(gallery):
    with gallery["lock"]:
        with open(GALLERY_PATH, "w") as f:
            json.dump({k: v.tolist() for k, v in gallery["data"].items()}, f)


# ── Frame Helpers ──────────────────────────────────────────────────────────────

def prepare_spoof_crop(frame, bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2-x1, y2-y1
    H, W = frame.shape[:2]
    m    = DEFAULT_CFG["spoof_crop_margin"]
    mx, my = int(w*m), int(h*m)
    crop = frame[max(0,y1-my):min(H,y2+my), max(0,x1-mx):min(W,x2+mx)]
    return crop if crop.size else None


def fft_tensor(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (80, 80))
    mag  = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)
    mag  = (mag - mag.mean()) / (mag.std() + 1e-8)
    return torch.from_numpy(mag.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def check_quality(frame, bbox):
    if bbox is None: return False, "No face"
    x1, y1, x2, y2 = bbox
    w, h = x2-x1, y2-y1
    if w < DEFAULT_CFG["min_face_px"] or h < DEFAULT_CFG["min_face_px"]:
        return False, f"Move closer ({w}x{h}px)"
    crop = frame[max(0,y1):y2, max(0,x1):x2]
    if crop.size == 0: return False, "Bad crop"
    gray      = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    bright    = gray.mean()
    if sharpness < DEFAULT_CFG["sharpness_min"]: return False, "Blurry"
    if bright < DEFAULT_CFG["brightness_min"]:   return False, "Too dark"
    if bright > DEFAULT_CFG["brightness_max"]:   return False, "Too bright"
    return True, "Good"


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_box(img, bbox, label, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    ly = y1-32 if y1 > 38 else y2+6
    cv2.rectangle(img, (x1, ly), (x1+tw+8, ly+th+8), color, -1)
    cv2.putText(img, label, (x1+4, ly+th+2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)


def draw_bar(img, bbox, score, color):
    x1, y1, x2, y2 = bbox
    fill = int((x2-x1) * min(max(score, 0.0), 1.0))
    cv2.rectangle(img, (x1, y2+4), (x2,       y2+12), (50, 50, 50), -1)
    cv2.rectangle(img, (x1, y2+4), (x1+fill,  y2+12), color,        -1)


def draw_hud(img, raw, smooth, state, warming, thr):
    h, w = img.shape[:2]
    by   = h - 30
    cv2.rectangle(img, (0, by), (w, h), (20, 20, 20), -1)
    cv2.rectangle(img, (0, by+2),  (int(w*min(max(raw,   0), 1)), by+9),  (70,70,70), -1)
    c = (0,200,255) if warming else (0,220,0) if state == "real" else (0,0,220)
    cv2.rectangle(img, (0, by+10), (int(w*min(max(smooth, 0), 1)), by+18), c, -1)
    cv2.line(img, (int(w*thr), by+2), (int(w*thr), by+18), (255,255,255), 2)
    txt = "Warming up..." if warming else \
          f"raw={raw:.3f}  smooth={smooth:.3f}  thr={thr:.3f}  -> {state.upper()}"
    cv2.putText(img, txt, (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, c, 1)


def draw_attendance_panel(img, attendance):
    ox = img.shape[1] - 240
    cv2.putText(img, "Attendance", (ox, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    for i, (name, t) in enumerate(attendance.items()):
        cv2.putText(img, f"{name}  {t[-8:]}", (ox, 44+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,210,0), 1)


# ── Video Processor ────────────────────────────────────────────────────────────

class AttendanceProcessor(VideoProcessorBase):
    """Runs in a background thread; communicates with Streamlit via public attrs."""

    def __init__(self):
        self.device, self.embed_mtcnn, self.display_mtcnn, \
            self.embedder, self.spoof_net, spoof_default_thr = load_models()
        self.gallery = get_gallery()

        # Config — written by Streamlit main thread, read by processor thread.
        # Simple attribute assignment is atomic in CPython (GIL-safe for scalars).
        self.rec_thr    = DEFAULT_CFG["recognition_threshold"]
        self.spoof_thr  = spoof_default_thr
        self.mode       = "recognize"   # "recognize" | "enroll"
        self.enroll_name   = ""
        self.enroll_target = DEFAULT_CFG["enrollment_frames"]
        self.reset_enroll  = False      # set True to discard collected frames

        # Internal recognition state
        self._emb_buf     = deque(maxlen=DEFAULT_CFG["embedding_buffer_size"])
        self._stable_name = "..."
        self._stable_score = 0.0
        self._candidate   = None
        self._cand_run    = 0

        # Internal spoof state
        self._score_buf   = deque(maxlen=DEFAULT_CFG["spoof_buffer_size"])
        self._consec_r    = 0
        self._consec_s    = 0
        self._spoof_state = "real"

        # Enrollment state
        self._enroll_embs  = []
        self._enroll_pidx  = 0
        self._enroll_ptick = time.time()

        # Attendance for this session
        self._attendance = {}

        # Latest status — read by Streamlit main thread (dict replace is atomic)
        self.latest_status: dict = {}

        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, "w", newline="") as f:
                csv.writer(f).writerow(["name", "date", "time"])

    # ── Inference helpers ──────────────────────────────────────

    @torch.no_grad()
    def _detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face_t, prob = self.embed_mtcnn(rgb, return_prob=True)
        if face_t is None or prob < 0.90:
            return None, None, None
        boxes, _ = self.display_mtcnn.detect(rgb)
        if boxes is None:
            return face_t, None, None
        bbox = boxes[0].astype(int)
        crop = prepare_spoof_crop(bgr, bbox)
        return face_t, bbox, crop

    @torch.no_grad()
    def _embed(self, face_t):
        return F.normalize(
            self.embedder(face_t.unsqueeze(0).to(self.device)), p=2, dim=1
        ).squeeze(0).cpu()

    @torch.no_grad()
    def _spoof(self, face_bgr):
        rgb = SPOOF_TRANSFORM(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        fft = fft_tensor(face_bgr).to(self.device)
        raw = torch.sigmoid(self.spoof_net(rgb, fft)).item()
        self._score_buf.append(raw)
        smooth  = sum(self._score_buf) / len(self._score_buf)
        warming = len(self._score_buf) < DEFAULT_CFG["spoof_buffer_size"]
        thr = self.spoof_thr
        if smooth >= thr: self._consec_r += 1; self._consec_s  = 0
        else:             self._consec_s += 1; self._consec_r  = 0
        if self._consec_r >= DEFAULT_CFG["spoof_real_confirm_frames"]:
            self._spoof_state = "real"
        elif self._consec_s >= DEFAULT_CFG["spoof_confirm_frames"]:
            self._spoof_state = "spoof"
        return self._spoof_state == "real", raw, smooth, warming

    def _recognize(self, emb):
        with self.gallery["lock"]:
            g = dict(self.gallery["data"])
        if not g:
            return "Unknown", 0.0
        scores = {n: F.cosine_similarity(emb.unsqueeze(0), e.unsqueeze(0)).item()
                  for n, e in g.items()}
        best, sc = max(scores.items(), key=lambda x: x[1])
        return (best, sc) if sc >= self.rec_thr else ("Unknown", sc)

    def _reset_recognition(self):
        self._emb_buf.clear()
        self._stable_name  = "..."; self._stable_score = 0.0
        self._candidate    = None;  self._cand_run     = 0
        self._score_buf.clear()
        self._consec_r = self._consec_s = 0
        self._spoof_state = "real"

    # ── Main frame callback ────────────────────────────────────

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img  = frame.to_ndarray(format="bgr24")
        disp = img.copy()

        if self.reset_enroll:
            self._enroll_embs  = []
            self._enroll_pidx  = 0
            self._enroll_ptick = time.time()
            self.reset_enroll  = False

        face_t, bbox, spoof_crop = self._detect(img)

        if self.mode == "enroll":
            self._do_enroll(disp, img, face_t, bbox)
        else:
            self._do_recognize(disp, face_t, bbox, spoof_crop)

        # Publish status atomically (dict replace)
        self.latest_status = {
            "mode":          self.mode,
            "stable_name":   self._stable_name,
            "stable_score":  self._stable_score,
            "attendance":    dict(self._attendance),
            "enroll_count":  len(self._enroll_embs),
            "enroll_target": self.enroll_target,
            "enroll_name":   self.enroll_name,
        }

        return av.VideoFrame.from_ndarray(disp, format="bgr24")

    # ── Enrollment logic ───────────────────────────────────────

    def _do_enroll(self, disp, img, face_t, bbox):
        name = self.enroll_name
        n    = len(self._enroll_embs)
        tgt  = self.enroll_target

        # Progress bar
        bw = 300
        cv2.rectangle(disp, (10, 10), (10+bw, 32), (40,40,40), -1)
        cv2.rectangle(disp, (10, 10), (10+int(bw*n/tgt), 32), (0,210,0), -1)
        cv2.putText(disp, f"Enrolling: {name}  {n}/{tgt}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if time.time() - self._enroll_ptick > 2.5:
            self._enroll_pidx  = (self._enroll_pidx + 1) % len(PROMPTS)
            self._enroll_ptick = time.time()
        cv2.putText(disp, PROMPTS[self._enroll_pidx], (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,230,230), 2)

        if bbox is not None:
            ok, msg = check_quality(img, bbox)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,210,0) if ok else (0,60,220), 2)
            cv2.putText(disp, msg, (10, disp.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0,210,0) if ok else (0,80,220), 2)

            if ok and face_t is not None and n < tgt:
                self._enroll_embs.append(self._embed(face_t))

                if len(self._enroll_embs) >= tgt:
                    mean_emb = F.normalize(
                        torch.stack(self._enroll_embs).mean(0), p=2, dim=0)
                    with self.gallery["lock"]:
                        self.gallery["data"][name] = mean_emb
                    save_gallery(self.gallery)
                    self._enroll_embs = []
                    self.mode = "recognize"
                    self._reset_recognition()

    # ── Recognition logic ──────────────────────────────────────

    def _do_recognize(self, disp, face_t, bbox, spoof_crop):
        if face_t is None or spoof_crop is None:
            self._reset_recognition()
            return

        is_real, raw, smooth, warming = self._spoof(spoof_crop)

        if warming:
            if bbox is not None:
                buf  = len(self._score_buf)
                bmax = DEFAULT_CFG["spoof_buffer_size"]
                pct  = int((bbox[2]-bbox[0]) * buf / bmax)
                cv2.rectangle(disp, (bbox[0], bbox[1]-8), (bbox[2], bbox[1]), (40,40,40), -1)
                cv2.rectangle(disp, (bbox[0], bbox[1]-8), (bbox[0]+pct, bbox[1]), (0,180,255), -1)
                cv2.rectangle(disp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,180,255), 2)
                cv2.putText(disp, f"Checking {buf}/{bmax}", (bbox[0], bbox[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,180,255), 1)

        elif not is_real:
            self._emb_buf.clear()
            self._stable_name = "..."; self._candidate = None; self._cand_run = 0
            if bbox is not None:
                draw_box(disp, bbox, f"SPOOF  {smooth:.2f}", (0,0,255))
                draw_bar(disp, bbox, smooth, (0,0,255))

        else:
            self._emb_buf.append(self._embed(face_t))
            avg = F.normalize(torch.stack(list(self._emb_buf)).mean(0), p=2, dim=0)
            rname, rscore = self._recognize(avg)

            if rname == self._candidate:
                self._cand_run += 1
            else:
                self._candidate = rname; self._cand_run = 1

            if self._cand_run >= DEFAULT_CFG["label_stable_frames"]:
                self._stable_name  = self._candidate
                self._stable_score = rscore

            if bbox is not None:
                known = self._stable_name not in ("Unknown", "...")
                col   = (0,200,0) if known else (130,130,130)
                label = f"LIVE {smooth:.2f} | {self._stable_name} {self._stable_score:.2f}"
                draw_box(disp, bbox, label, col)
                draw_bar(disp, bbox, self._stable_score, col)

                if known and self._stable_name not in self._attendance:
                    now = datetime.now()
                    d, t = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
                    self._attendance[self._stable_name] = f"{d} {t}"
                    with open(LOG_PATH, "a", newline="") as f:
                        csv.writer(f).writerow([self._stable_name, d, t])

        draw_hud(disp, raw, smooth, "real" if is_real else "spoof", warming, self.spoof_thr)
        draw_attendance_panel(disp, self._attendance)


# ── Streamlit UI ───────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Face Attendance System",
        page_icon="🎭",
        layout="wide",
    )

    st.title("Face Recognition Attendance System")

    # ── Session state defaults ─────────────────────────────────
    if "enroll_active" not in st.session_state:
        st.session_state.enroll_active     = False
        st.session_state.enroll_name       = ""
        st.session_state.start_enroll_flag = False

    # ── Load models (cached — only runs once) ─────────────────
    with st.spinner("Loading models... (first run only)"):
        load_models()

    gallery = get_gallery()

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")

        with gallery["lock"]:
            enrolled = list(gallery["data"].keys())

        if enrolled:
            st.success(f"Enrolled ({len(enrolled)}): {', '.join(enrolled)}")
        else:
            st.warning("No one enrolled yet")

        st.divider()

        tab_rec, tab_enroll, tab_del = st.tabs(["Recognize", "Enroll", "Delete"])

        with tab_rec:
            st.info("Camera recognizes enrolled persons automatically when started.")

        with tab_enroll:
            name_input = st.text_input("Person's name")
            btn_enroll = st.button(
                "Start Enrollment",
                disabled=not name_input.strip() or st.session_state.enroll_active,
            )
            if btn_enroll and name_input.strip():
                st.session_state.enroll_active     = True
                st.session_state.enroll_name       = name_input.strip()
                st.session_state.start_enroll_flag = True

            if st.session_state.enroll_active:
                st.info(f"Enrolling **{st.session_state.enroll_name}** — look at the camera.")

        with tab_del:
            if enrolled:
                del_name = st.selectbox("Select person", enrolled)
                if st.button("Delete", type="primary"):
                    with gallery["lock"]:
                        gallery["data"].pop(del_name, None)
                    save_gallery(gallery)
                    st.success(f"Deleted '{del_name}'")
                    st.rerun()
            else:
                st.info("Gallery is empty.")

        st.divider()
        st.subheader("Thresholds")
        _, _, _, _, _, default_spoof_thr = load_models()
        rec_thr   = st.slider("Recognition", 0.50, 1.00,
                               DEFAULT_CFG["recognition_threshold"], 0.01)
        spoof_thr = st.slider("Liveness",    0.10, 0.90,
                               float(default_spoof_thr), 0.01)

    # ── Main area ──────────────────────────────────────────────
    col_vid, col_info = st.columns([3, 1])

    with col_vid:
        ctx = webrtc_streamer(
            key="attendance",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=AttendanceProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_info:
        st.subheader("Live Status")
        ph_name   = st.empty()
        ph_score  = st.empty()
        ph_enroll = st.empty()
        st.divider()
        st.subheader("Session Attendance")
        ph_att = st.empty()

    # ── Sync processor config ──────────────────────────────────
    if ctx.video_processor:
        proc = ctx.video_processor

        # Push config
        proc.rec_thr   = rec_thr
        proc.spoof_thr = spoof_thr

        # Handle enrollment state
        if st.session_state.start_enroll_flag:
            proc.enroll_name   = st.session_state.enroll_name
            proc.enroll_target = DEFAULT_CFG["enrollment_frames"]
            proc.reset_enroll  = True
            proc.mode          = "enroll"
            st.session_state.start_enroll_flag = False
        elif st.session_state.enroll_active:
            proc.mode        = "enroll"
            proc.enroll_name = st.session_state.enroll_name
        else:
            proc.mode = "recognize"

        # Read latest status
        status = proc.latest_status
        if status:
            ph_name.metric("Detected", status.get("stable_name", "..."))
            ph_score.metric("Score", f"{status.get('stable_score', 0.0):.3f}")

            # Enrollment completed when processor flips back to recognize
            if st.session_state.enroll_active and status.get("mode") == "recognize":
                st.session_state.enroll_active = False
                st.toast(f"Enrolled '{status['enroll_name']}' successfully!", icon="✅")
                st.rerun()

            if st.session_state.enroll_active:
                cnt = status.get("enroll_count", 0)
                tgt = status.get("enroll_target", DEFAULT_CFG["enrollment_frames"])
                ph_enroll.progress(cnt / tgt, text=f"{cnt}/{tgt} frames captured")

            att = status.get("attendance", {})
            if att:
                ph_att.dataframe(
                    pd.DataFrame(list(att.items()), columns=["Name", "Time"]),
                    use_container_width=True, hide_index=True,
                )
            else:
                ph_att.info("Waiting for recognition...")

    # ── Full Attendance Log ────────────────────────────────────
    st.divider()
    with st.expander("Full Attendance Log", expanded=False):
        if os.path.exists(LOG_PATH):
            df = pd.read_csv(LOG_PATH)
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv_bytes = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv_bytes, "attendance_log.csv", "text/csv")
        else:
            st.info("No log recorded yet.")


if __name__ == "__main__":
    main()
