
import cv2
import os
import time
import numpy as np
from facenet_pytorch import MTCNN
import torch
from pathlib import Path

DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_ROOT           = "./my_spoof_data"
TARGET_PER_SCENARIO = 80    # frames per person per real scenario
TARGET_FAKE         = 120   # frames per fake scenario (shared across people)


# ── Camera ────────────────────────────────────────────────────────────────────

def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return cap


# ── Core collector ────────────────────────────────────────────────────────────

def collect_session(label, scenario_name, save_dir, mtcnn,
                    target, person_name=""):
    """
    Runs a single recording session in a cv2 window.

    label         : "real" or "fake"
    scenario_name : e.g. "bright_smile", "phone_screen"
    save_dir      : destination folder (created if needed)
    target        : number of frames to collect
    person_name   : shown in window title (empty for fake sessions)
    """
    os.makedirs(save_dir, exist_ok=True)
    existing = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
    already_have = existing

    cap  = open_camera()
    idx  = existing
    auto = False

    who  = f" [{person_name}]" if person_name else ""
    title = f"[{label.upper()}]{who} — {scenario_name} | A=auto  SPACE=snap  Q=done"
    print(f"\n  {'─'*54}")
    print(f"  Recording [{label.upper()}]{who} — {scenario_name}")
    print(f"  Target: {target} | Already in folder: {existing}")
    print(f"  {'─'*54}")

    last_capture_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb              = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs     = mtcnn.detect(rgb)
        display          = frame.copy()
        saved_this_round = idx - already_have
        has_face         = boxes is not None and probs[0] > 0.85
        crop             = None

        if has_face:
            b   = boxes[0].astype(int)
            H, W = frame.shape[:2]
            x1,y1 = max(0,b[0]), max(0,b[1])
            x2,y2 = min(W,b[2]), min(H,b[3])
            mx = int((x2-x1)*0.2); my = int((y2-y1)*0.2)
            cx1 = max(0,x1-mx);  cy1 = max(0,y1-my)
            cx2 = min(W,x2+mx);  cy2 = min(H,y2+my)
            crop = frame[cy1:cy2, cx1:cx2]

            color = (0,200,0) if label == "real" else (0,0,220)
            cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)

            # Auto capture (rate-limited)
            now = time.time()
            if auto and crop.size > 0 and (now - last_capture_time) > 0.08:
                fname = os.path.join(save_dir,
                                     f"{scenario_name}_{idx:05d}.jpg")
                cv2.imwrite(fname, crop)
                idx += 1
                last_capture_time = now

        # ── HUD ──────────────────────────────────────────────────────────────
        bar_w    = 320
        progress = min(saved_this_round / target, 1.0)
        cv2.rectangle(display, (10,10), (10+bar_w, 30), (40,40,40), -1)
        bar_color = (0,200,0) if label=="real" else (0,0,200)
        cv2.rectangle(display, (10,10),
                      (10+int(bar_w*progress), 30), bar_color, -1)

        lbl_color = (0,220,0) if label=="real" else (60,60,220)
        who_str   = f"  {person_name}" if person_name else ""
        cv2.putText(display,
                    f"[{label.upper()}]{who_str} — {scenario_name}",
                    (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, lbl_color, 2)
        cv2.putText(display,
                    f"Saved: {saved_this_round}/{target}  "
                    f"({'AUTO' if auto else 'manual'})",
                    (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        if not has_face:
            cv2.putText(display, "No face — move into frame",
                        (10,110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,220), 2)

        if saved_this_round >= target:
            cv2.putText(display, "TARGET REACHED — press Q to continue",
                        (10,140), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0,220,220), 2)

        cv2.putText(display,
                    "A=auto  SPACE=snap  Q=done",
                    (10, display.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,160,160), 1)

        cv2.imshow(title, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('a'):
            auto = not auto
            print(f"  Auto-capture: {'ON' if auto else 'OFF'}")
        elif key == ord(' ') and has_face and crop is not None \
                and crop.size > 0:
            fname = os.path.join(save_dir,
                                 f"{scenario_name}_{idx:05d}.jpg")
            cv2.imwrite(fname, crop)
            idx += 1
            print(f"  Saved frame {idx}")

    cap.release()
    cv2.destroyAllWindows()
    collected = idx - already_have
    print(f"  ✅ Collected {collected} frames")
    return collected


# ── Scenarios ─────────────────────────────────────────────────────────────────

# Real scenarios: run for EACH person
REAL_SCENARIOS = [
    ("normal",
     "Sit naturally. Neutral expression. Normal room lighting."),
    ("bright_light",
     "Face a bright window or turn on extra lights. Neutral expression."),
    ("laughing",
     "Laugh naturally — open mouth, bright expression."),
    ("smiling_bright",
     "Big smile in bright lighting."),
    ("side_angle",
     "Slowly turn head left and right 30-45° while recording."),
    ("glasses",
     "Wear glasses if you have them. Neutral expression."),
    ("dim_light",
     "Dim the room lights. Normal expression."),
]

# Fake scenarios: recorded ONCE and shared across all people
FAKE_SCENARIOS = [
    ("phone_screen",
     "Hold your phone showing a photo of any enrolled person towards the camera."),
    ("phone_video",
     "Play a video of any enrolled person on your phone. Hold it to camera."),
    ("printed_photo",
     "Print a photo of any enrolled person. Hold at different angles."),
    ("laptop_screen",
     "Open a photo on another laptop screen. Hold towards camera."),
]


# ── Dataset summary ───────────────────────────────────────────────────────────

def print_summary():
    real_dir = os.path.join(SAVE_ROOT, "real")
    fake_dir = os.path.join(SAVE_ROOT, "fake")

    real_total = 0
    if os.path.exists(real_dir):
        for person_dir in sorted(Path(real_dir).iterdir()):
            if person_dir.is_dir():
                n = len(list(person_dir.rglob("*.jpg")))
                print(f"    real/{person_dir.name:<20}  {n} frames")
                real_total += n

    fake_total = 0
    if os.path.exists(fake_dir):
        for d in sorted(Path(fake_dir).iterdir()):
            if d.is_dir():
                n = len(list(d.rglob("*.jpg")))
                print(f"    fake/{d.name:<20}  {n} frames")
                fake_total += n

    print(f"\n    Total real : {real_total}")
    print(f"    Total fake : {fake_total}")
    print(f"    Grand total: {real_total + fake_total}")
    return real_total, fake_total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mtcnn = MTCNN(min_face_size=60, keep_all=False,
                  post_process=False, device=DEVICE)

    print("\n" + "="*58)
    print("  Anti-Spoofing Multi-Person Data Collector")
    print("="*58)

    # ── Step 1: register people ───────────────────────────────────────────────
    print("""
Each person sits in front of the camera one at a time.
You will go through several real-face scenarios per person.
Fake (spoof) scenarios are collected once at the end.
""")

    people = []
    print("Enter the names of people who will participate.")
    print("Press ENTER with no name when done.\n")
    while True:
        name = input("  Person name (or ENTER to finish): ").strip()
        if not name:
            if not people:
                print("  Need at least one person.")
                continue
            break
        # Sanitise for folder name
        safe = name.lower().replace(" ", "_")
        people.append((name, safe))
        print(f"  Added: {name}")

    print(f"\n  People registered: {[n for n,_ in people]}")

    # ── Step 2: real scenarios per person ─────────────────────────────────────
    for display_name, safe_name in people:
        person_dir = os.path.join(SAVE_ROOT, "real", f"person_{safe_name}")
        os.makedirs(person_dir, exist_ok=True)

        print(f"\n{'='*58}")
        print(f"  NOW RECORDING: {display_name}")
        print(f"  Please sit in front of the camera.")
        print(f"{'='*58}")
        input("  Press ENTER when ready → ")

        for scenario, description in REAL_SCENARIOS:
            print(f"\n  Scenario : {scenario}")
            print(f"  Instructions : {description}")
            resp = input("  ENTER to start, 'skip' to skip → ").strip().lower()
            if resp == "skip":
                print("  Skipped.")
                continue

            collect_session(
                label         = "real",
                scenario_name = scenario,
                save_dir      = person_dir,
                mtcnn         = mtcnn,
                target        = TARGET_PER_SCENARIO,
                person_name   = display_name,
            )

        # Per-person summary
        n = len(list(Path(person_dir).rglob("*.jpg")))
        print(f"\n  ✅ {display_name} done — {n} real frames saved")

    # ── Step 3: fake scenarios (once, shared) ─────────────────────────────────
    print(f"\n{'='*58}")
    print("  FAKE (SPOOF) SCENARIOS")
    print("  These are recorded once and apply to all people.")
    print(f"{'='*58}")
    input("  Press ENTER to begin → ")

    fake_dir = os.path.join(SAVE_ROOT, "fake", "shared")
    os.makedirs(fake_dir, exist_ok=True)

    for scenario, description in FAKE_SCENARIOS:
        print(f"\n  Scenario : {scenario}")
        print(f"  Instructions : {description}")
        resp = input("  ENTER to start, 'skip' to skip → ").strip().lower()
        if resp == "skip":
            print("  Skipped.")
            continue

        collect_session(
            label         = "fake",
            scenario_name = scenario,
            save_dir      = fake_dir,
            mtcnn         = mtcnn,
            target        = TARGET_FAKE,
            person_name   = "",
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print("  Collection complete — dataset summary:")
    print(f"{'='*58}")
    real_total, fake_total = print_summary()

    print()
    if real_total < 200 or fake_total < 100:
        print("  ⚠️  Dataset is small — more data = better fine-tuning.")
        print("     Run collect_data.py again to add more people or scenarios.")
    else:
        print("  ✅ Good dataset size. Ready to fine-tune:")
        print("     python finetune.py")


if __name__ == "__main__":
    main()
