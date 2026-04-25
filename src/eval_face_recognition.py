"""
Face recognition evaluation (project-native, no LFW dependency).

Evaluates a MobileFaceNet embedding checkpoint using gallery-vs-probe matching
with cosine similarity on ImageFolder identities (default: dataset/vggface2_train).

Outputs:
  - stdout summary
  - checkpoints/face_recognition_evaluation_report.txt

Usage:
  python src/eval_face_recognition.py
  python src/eval_face_recognition.py --max-identities 120 --gallery-per-id 3 --probe-per-id 5
"""

import argparse
import os
import random
import time
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from torchvision import transforms

from mobilefacenet import MobileFaceNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_identity_index(data_dir: str):
    identity_to_paths = {}
    for identity in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, identity)
        if not os.path.isdir(p):
            continue
        images = []
        for name in sorted(os.listdir(p)):
            low = name.lower()
            if low.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                images.append(os.path.join(p, name))
        if images:
            identity_to_paths[identity] = images
    return identity_to_paths


@torch.no_grad()
def embed_image(model, tf, path: str, device: torch.device):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    emb = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    emb = F.normalize(emb, p=2, dim=1).squeeze(0).cpu()
    return emb, (t1 - t0) * 1000.0


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate face recognition checkpoint")
    parser.add_argument("--data-dir", default="./dataset/vggface2_train")
    parser.add_argument("--checkpoint", default="./checkpoints/best_model.pth")
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--min-images-per-id", type=int, default=8)
    parser.add_argument("--gallery-per-id", type=int, default=3)
    parser.add_argument("--probe-per-id", type=int, default=5)
    parser.add_argument("--max-identities", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-out", default="./checkpoints/face_recognition_evaluation_report.txt")
    parser.add_argument("--threshold", type=float, default=0.55)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    id_to_paths = build_identity_index(args.data_dir)
    eligible = [
        k for k, v in id_to_paths.items()
        if len(v) >= max(args.min_images_per_id, args.gallery_per_id + args.probe_per_id)
    ]

    if not eligible:
        raise RuntimeError("No eligible identities found for requested gallery/probe split.")

    rng = np.random.default_rng(args.seed)
    if len(eligible) > args.max_identities:
        eligible = sorted(rng.choice(eligible, size=args.max_identities, replace=False).tolist())

    model = MobileFaceNet(embedding_size=args.embedding_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError("Expected key 'model_state_dict' in checkpoint")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    gallery_vectors = {}
    probes = []
    latencies_ms = []

    for identity in eligible:
        paths = id_to_paths[identity]
        idx = rng.permutation(len(paths))
        paths = [paths[i] for i in idx]

        gallery_paths = paths[: args.gallery_per_id]
        probe_paths = paths[args.gallery_per_id : args.gallery_per_id + args.probe_per_id]

        gal_embs = []
        for p in gallery_paths:
            emb, latency = embed_image(model, tf, p, device)
            gal_embs.append(emb)
            latencies_ms.append(latency)

        gallery_vectors[identity] = F.normalize(torch.stack(gal_embs).mean(dim=0), p=2, dim=0)

        for p in probe_paths:
            probes.append((identity, p))

    y_true = []
    y_pred = []
    genuine_scores = []
    impostor_scores = []

    with torch.no_grad():
        for true_id, probe_path in probes:
            emb, latency = embed_image(model, tf, probe_path, device)
            latencies_ms.append(latency)

            best_id = None
            best_score = -1.0
            true_score = None
            best_impostor_score = -1.0

            for candidate_id, center in gallery_vectors.items():
                score = F.cosine_similarity(emb.unsqueeze(0), center.unsqueeze(0)).item()
                if candidate_id == true_id:
                    true_score = score
                else:
                    if score > best_impostor_score:
                        best_impostor_score = score
                if score > best_score:
                    best_score = score
                    best_id = candidate_id

            y_true.append(true_id)
            y_pred.append(best_id)
            if true_score is not None:
                genuine_scores.append(true_score)
            if best_impostor_score > -1.0:
                impostor_scores.append(best_impostor_score)

    acc = accuracy_score(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, digits=4)

    report_lines = [
        "=" * 70,
        "FACE RECOGNITION EVALUATION REPORT",
        "=" * 70,
        "",
        f"Checkpoint        : {args.checkpoint}",
        f"Dataset           : {args.data_dir}",
        f"Device            : {device}",
        f"Identities used   : {len(eligible)}",
        f"Gallery / Identity: {args.gallery_per_id}",
        f"Probe / Identity  : {args.probe_per_id}",
        f"Total probes      : {len(probes)}",
        "",
        "RESULTS",
        "-" * 70,
        f"Top-1 Accuracy    : {acc * 100:.2f}%",
        f"Threshold         : {args.threshold}",
        f"Mean Inference    : {mean(latencies_ms):.2f} ms / face",
        f"Median Inference  : {float(np.median(latencies_ms)):.2f} ms / face",
        "",
        "Classification Report:",
        cls_report,
    ]

    report = "\n".join(report_lines)
    print(report)

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved: {args.report_out}")

    threshold = args.threshold
    _, ax = plt.subplots(figsize=(8, 5))
    ax.hist(genuine_scores, bins=50, alpha=0.7, color="green", label=f"Genuine ({len(genuine_scores)})")
    ax.hist(impostor_scores, bins=50, alpha=0.7, color="red", label=f"Impostor ({len(impostor_scores)})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Count")
    ax.set_title("Face Recognition — Match Score Distribution")
    ax.legend()
    plt.tight_layout()
    plot_out = os.path.join(os.path.dirname(args.report_out), "face_recognition_score_distribution.png")
    plt.savefig(plot_out, dpi=150)
    plt.close()
    print(f"Score distribution saved: {plot_out}")


if __name__ == "__main__":
    main()
