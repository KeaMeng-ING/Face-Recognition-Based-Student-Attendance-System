# Face Recognition-Based Student Attendance System with Anti-Spoofing Detection

**ITM-390 Machine Learning — American University of Phnom Penh**

A real-time student attendance system that combines face recognition and anti-spoofing detection. The system verifies that a presented face is live (not a photo or screen replay), identifies the student, and automatically logs their attendance to a CSV file.

---

## Team

| Role | Name | Student ID |
|---|---|---|
| Team Leader | ING Kea Meng | 2023484 |
| Member | NANG Vannet | 2024024 |
| Member | UNG Seangeang | 2024471 |

Advisor: **Prof. Kuntha PIN**

---

## How It Works

The pipeline enforces a strict two-gate model:

```
Camera → Face Detection (MTCNN) → Anti-Spoofing → Face Recognition → Attendance CSV
                                        ↓ Fake              ↓ Unknown
                                     Rejected             Rejected
```

1. **Face Detection** — MTCNN detects and crops the face from each video frame
2. **Anti-Spoofing** — AntiSpoofNet classifies the face as Real or Fake (rejects photos and screen replays)
3. **Face Recognition** — MobileFaceNet embeds the face and matches it against the enrolled student gallery via cosine similarity
4. **Attendance Recording** — Confirmed students are logged to `attendance_log.csv` with name, date, and time

---

## Models

### Face Recognition — MobileFaceNet + ArcFace
- 512-dimensional L2-normalised face embeddings
- Trained from scratch on VGGFace2 dataset (197,693 images, 540 identities)
- ArcFace loss (scale=30, margin=0.30)
- Inference: cosine similarity against a gallery of per-student mean embeddings

### Anti-Spoofing — Dual-Branch AntiSpoofNet
- **RGB branch**: MobileNetV2 backbone (ImageNet pre-trained, layers 0–5 frozen) → 1,280-dim feature
- **FFT branch**: 3-stage CNN on the log-magnitude spectrum of an 80×80 grayscale crop → 2,304-dim feature
- Both branches concatenated (3,584-dim) → 3-layer classifier with sigmoid output
- Trained on NUAA dataset + custom-collected webcam data

---

## Project Structure

```
demo/
├── src/
│   ├── app.py                  # Main application (camera + anti-spoofing + recognition)
│   ├── main.py                 # Entry point
│   ├── mobilefacenet.py        # MobileFaceNet architecture
│   ├── train.py                # Face recognition training
│   ├── train_antispoofing.py   # Anti-spoofing training
│   ├── eval_antispoofing.py    # Anti-spoofing evaluation
│   ├── eval_face_recognition.py
│   ├── evaluate_models.py
│   ├── nuaa_preprocess.py      # NUAA dataset preprocessing (FFT feature extraction)
│   ├── collect_data.py         # Collect personal real/fake samples
│   ├── collect_screen_attacks.py
│   └── finetune_webcam.py      # Fine-tune on webcam data
├── checkpoints/
│   ├── best_model.pth              # Face recognition checkpoint
│   └── best_model_antispoofing.pth # Anti-spoofing checkpoint
├── dataset/
│   ├── NUAA/                   # NUAA anti-spoofing dataset
│   ├── vggface2_train/         # VGGFace2 face recognition dataset
│   ├── processed_dataset/      # Preprocessed NUAA (FFT features + CSV index)
│   └── webcam_data/            # Custom collected webcam samples
├── my_spoof_data/              # Personal screen-attack samples
├── face_gallery.json           # Enrolled student embeddings
├── attendance_log.csv          # Attendance records output
└── requirements.txt
```

---

## Installation

```bash
# Clone the repository and enter the project folder
cd demo

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.x, OpenCV 4.7+

---

## Usage

### 1. Run the attendance system

```bash
python src/main.py
```

The menu options:

```
1 — Enroll new person
2 — Start attendance (camera + detection)
3 — Delete person
4 — Adjust recognition threshold
5 — Adjust liveness threshold
6 — Quit
```

### 2. Enroll a student
Select option `1`, enter the student's name, and follow the on-screen pose prompts. The system captures 20 good-quality frames automatically.

### 3. Start attendance
Select option `2`. The camera window shows:
- **Green box + LIVE** — real face, student recognised
- **Red box + SPOOF** — spoof attempt detected, rejected
- Bottom bar — live anti-spoofing score and threshold

Attendance is saved automatically to `attendance_log.csv`.

---

## Training

### Face Recognition

```bash
python src/train.py --data-dir dataset/vggface2_train --epochs 30
```

| Parameter | Value |
|---|---|
| Model | MobileFaceNet |
| Loss | ArcFace (s=30, m=0.30) |
| Optimizer | Adam (lr=0.0001) |
| Batch size | 64 |
| Epochs | 30 |
| Train/Val split | 80% / 20% |

### Anti-Spoofing

```bash
# Step 1 — Preprocess NUAA dataset (extract FFT features)
python src/nuaa_preprocess.py

# Step 2 — Train
python src/train_antispoofing.py
```

| Parameter | Value |
|---|---|
| Architecture | Dual-branch AntiSpoofNet (MobileNetV2 + FFT CNN) |
| Loss | Focal Loss (α=0.25, γ=3.0) |
| Optimizer | Adam (lr=0.0001, weight_decay=1e-5) |
| LR Scheduler | CosineAnnealingLR |
| Batch size | 32 |
| Epochs | 60 (early stopping, patience=8) |
| Val split | Cross-subject — subjects 0001, 0003, 0008, 0015 held out |

---

## Collecting Custom Data

```bash
# Collect real face samples (webcam)
python src/collect_data.py

# Collect screen-replay attack samples
python src/collect_screen_attacks.py

# Fine-tune anti-spoofing model on new webcam data
python src/finetune_webcam.py
```

---

## Results

| Metric | Value | Target |
|---|---|---|
| Face Recognition Accuracy | 94% | >90% |
| Recognition Speed | 40 ms/face | <100 ms |
| False Positive Rate | <2% | <5% |
| Anti-Spoofing Accuracy | 97.55% | >95% |
| End-to-End Processing Time | 130 ms | <200 ms |

**Anti-Spoofing classification report (NUAA validation set):**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Real (Live) | 99.69% | 99.92% | 99.80% |
| Fake (Spoofed) | 99.64% | 98.59% | 99.12% |
| **Overall** | | | **99.68%** |

---

## Datasets

| Dataset | Purpose | Size |
|---|---|---|
| VGGFace2 | Face recognition training | 197,693 images, 540 identities |
| NUAA | Anti-spoofing training | 12,643 images |
| Webcam Data | Fine-tuning + screen attack | ~1,500 frames |
