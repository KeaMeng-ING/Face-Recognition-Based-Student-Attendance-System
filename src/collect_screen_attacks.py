"""
Screen Attack Data Collector
=============================
Dedicated collector for phone / tablet screen spoof attacks.
Saves to my_spoof_data/fake/screen_attacks/ alongside existing data.

Why a separate script?
  collect_data.py collects 120 frames per fake scenario.
  To compete with NUAA's ~7,400 fake samples you need many more
  screen frames across varied devices, distances, angles, and lighting.
  This script walks you through all combinations systematically.

Folder structure saved:
    my_spoof_data/
        fake/
            screen_attacks/
                phone_photo_00001.jpg       ← photo on phone, straight on
                phone_photo_angle_00001.jpg ← photo on phone, angled
                phone_video_00001.jpg       ← video on phone
                tablet_photo_00001.jpg      ← photo on tablet
                ...

Controls during recording:
    A      toggle auto-capture (~12 fps)
    SPACE  capture single frame
    Q      done with this scenario

Usage:
    python src/collect_screen_attacks.py
"""

import cv2
import os
import time
import torch
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR  = "./my_spoof_data/fake/screen_attacks"
TARGET    = 150     # frames per scenario — increase if you want more


# ── Camera ────────────────────────────────────────────────────────────────────

def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return cap


# ── Collector (same style as collect_data.py) ─────────────────────────────────

def collect_session(scenario_name, save_dir, mtcnn, target):
    os.makedirs(save_dir, exist_ok=True)
    existing = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
    already_have = existing

    cap  = open_camera()
    idx  = existing
    auto = False
    last_capture_time = 0.0

    title = f"[FAKE] {scenario_name} | A=auto  SPACE=snap  Q=done"
    print(f"\n  {'─'*54}")
    print(f"  Recording [FAKE] — {scenario_name}")
    print(f"  Target: {target} | Already in folder: {existing}")
    print(f"  {'─'*54}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(rgb)
        display      = frame.copy()
        saved        = idx - already_have
        has_face     = boxes is not None and probs[0] > 0.85
        crop         = None

        if has_face:
            b = boxes[0].astype(int)
            H, W = frame.shape[:2]
            x1, y1 = max(0, b[0]), max(0, b[1])
            x2, y2 = min(W, b[2]), min(H, b[3])
            mx = int((x2 - x1) * 0.2);  my = int((y2 - y1) * 0.2)
            cx1 = max(0, x1 - mx);  cy1 = max(0, y1 - my)
            cx2 = min(W, x2 + mx);  cy2 = min(H, y2 + my)
            crop = frame[cy1:cy2, cx1:cx2]

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 220), 2)

            now = time.time()
            if auto and crop.size > 0 and (now - last_capture_time) > 0.08:
                fname = os.path.join(save_dir, f"{scenario_name}_{idx:05d}.jpg")
                cv2.imwrite(fname, crop)
                idx += 1
                last_capture_time = now

        # ── HUD ──────────────────────────────────────────────────────────────
        bar_w    = 320
        progress = min(saved / target, 1.0)
        cv2.rectangle(display, (10, 10), (10 + bar_w, 30), (40, 40, 40), -1)
        cv2.rectangle(display, (10, 10),
                      (10 + int(bar_w * progress), 30), (0, 0, 200), -1)

        cv2.putText(display, f"[FAKE] {scenario_name}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 220), 2)
        cv2.putText(display, f"Saved: {saved}/{target}  ({'AUTO' if auto else 'manual'})",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if not has_face:
            cv2.putText(display, "No face detected on screen — adjust angle",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

        if saved >= target:
            cv2.putText(display, "TARGET REACHED — press Q to continue",
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2)

        cv2.putText(display, "A=auto  SPACE=snap  Q=done",
                    (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

        cv2.imshow(title, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('a'):
            auto = not auto
            print(f"  Auto-capture: {'ON' if auto else 'OFF'}")
        elif key == ord(' ') and has_face and crop is not None and crop.size > 0:
            fname = os.path.join(save_dir, f"{scenario_name}_{idx:05d}.jpg")
            cv2.imwrite(fname, crop)
            idx += 1
            print(f"  Saved frame {idx}")

    cap.release()
    cv2.destroyAllWindows()
    collected = idx - already_have
    print(f"  ✅ Collected {collected} frames")
    return collected


# ── Scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    # (scenario_name, instructions)

    # ── Phone — photo ─────────────────────────────────────────────────────────
    ("phone_photo",
     "Open a photo of ANY face on your phone.\n"
     "Hold phone straight at the camera. Normal distance (~40cm)."),

    ("phone_photo_close",
     "Same phone photo — move CLOSER (~20cm from camera)."),

    ("phone_photo_far",
     "Same phone photo — move FURTHER BACK (~60cm from camera)."),

    ("phone_photo_angle_left",
     "Same phone photo — tilt the phone to the LEFT ~30°."),

    ("phone_photo_angle_right",
     "Same phone photo — tilt the phone to the RIGHT ~30°."),

    ("phone_photo_tilt_up",
     "Same phone photo — tilt the phone UPWARD ~20°."),

    ("phone_photo_dim",
     "Dim the room lights. Hold phone photo at normal distance."),

    ("phone_photo_bright",
     "Turn on extra lights / face a window. Hold phone photo at normal distance."),

    ("phone_photo_low_brightness",
     "Lower your phone screen brightness to minimum. Hold at normal distance."),

    # ── Phone — video ─────────────────────────────────────────────────────────
    ("phone_video",
     "Play a VIDEO of any face on your phone.\n"
     "Hold phone straight at the camera. Normal distance."),

    ("phone_video_angle",
     "Same phone video — tilt phone left or right ~30° while recording."),

    ("phone_video_dim",
     "Dim the room lights. Play video on phone at normal distance."),

    # ── Tablet ────────────────────────────────────────────────────────────────
    ("tablet_photo",
     "Open a photo of any face on a TABLET.\n"
     "Hold tablet straight at the camera."),

    ("tablet_photo_angle",
     "Same tablet photo — tilt the tablet left or right ~30°."),

    ("tablet_video",
     "Play a VIDEO of any face on a TABLET. Hold straight at camera."),

    # ── Laptop / monitor ──────────────────────────────────────────────────────
    ("laptop_screen",
     "Open a photo of any face on a LAPTOP screen.\n"
     "Hold the laptop or sit so the camera faces the screen."),
]


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary():
    if not os.path.exists(SAVE_DIR):
        print("  No data collected yet.")
        return 0

    print(f"\n  Saved to: {SAVE_DIR}")
    total = 0
    by_scenario = {}
    for f in Path(SAVE_DIR).glob("*.jpg"):
        # Extract scenario name (everything except the last _NNNNN)
        parts = f.stem.rsplit("_", 1)
        scenario = parts[0] if len(parts) == 2 else f.stem
        by_scenario[scenario] = by_scenario.get(scenario, 0) + 1

    for scenario, count in sorted(by_scenario.items()):
        print(f"    {scenario:<30}  {count:>5} frames")
        total += count

    print(f"\n  Total screen attack frames: {total}")
    nuaa_fake = 7452
    print(f"  NUAA printed photo frames : {nuaa_fake}")
    pct = total / nuaa_fake * 100
    print(f"  Coverage vs NUAA          : {pct:.1f}%")
    if pct < 30:
        print("  ⚠️  Still low — aim for 3,000+ to match NUAA volume")
    elif pct < 70:
        print("  ⚠️  Getting there — keep collecting more scenarios")
    else:
        print("  ✅ Good coverage — model should generalise to screen attacks")
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mtcnn = MTCNN(min_face_size=60, keep_all=False,
                  post_process=False, device=DEVICE)

    print("\n" + "=" * 58)
    print("  Screen Attack Data Collector")
    print("=" * 58)
    print(f"""
Goal: collect {TARGET} frames per scenario across phones, tablets,
distances, angles, and lighting conditions.

You can use photos/videos of ANY person — it doesn't need to be
someone enrolled in the gallery. The model learns to detect the
SCREEN, not the face on it.

TIP: Use auto-capture (press A) and slowly vary angle/distance
     while recording to get natural variation per scenario.
""")

    print("  Current dataset status:")
    print_summary()

    print(f"\n  About to run {len(SCENARIOS)} scenarios × {TARGET} frames each.")
    print(f"  You can skip any scenario by typing 'skip' when prompted.")
    input("\n  Press ENTER to begin → ")

    total_collected = 0

    for i, (scenario, description) in enumerate(SCENARIOS, 1):
        print(f"\n  [{i}/{len(SCENARIOS)}] Scenario: {scenario}")
        print(f"  Instructions: {description}")
        resp = input("  ENTER to start, 'skip' to skip → ").strip().lower()
        if resp == "skip":
            print("  Skipped.")
            continue

        collected = collect_session(
            scenario_name = scenario,
            save_dir      = SAVE_DIR,
            mtcnn         = mtcnn,
            target        = TARGET,
        )
        total_collected += collected

    print(f"\n{'=' * 58}")
    print("  Collection complete — final summary:")
    print(f"{'=' * 58}")
    print_summary()

    print("\n  Next steps:")
    print("  1. python src/nuaa_preprocess.py   ← rebuild dataset CSV")
    print("  2. python src/train_antispoofing.py ← retrain with screen data")


if __name__ == "__main__":
    main()
