"""
eval_rotation_model.py
======================
Evaluates the trained rotation model (v3, MobileNetV3-Small) against
real coin images from 2026Data using manual crop labels.

For each evaluated folder it:
  1. Crops the coin using the manual label (same as preprocess_dataset_v2.py)
  2. Runs the rotation model on --samples images from that folder
  3. Applies the outlier-filtered circular mean to get a folder correction
  4. Saves a visual panel: raw crop | rotated result | angle annotation

Also produces a summary CSV with per-folder angle predictions and std-dev,
so you can quickly spot folders where the model is uncertain.

USAGE
-----
    python eval_rotation_model.py \\
        --raw-root  /path/to/2026Data \\
        --labels    /path/to/crop_labels.json \\
        --model     /path/to/models/rotation_v3.pth \\
        --out-dir   /path/to/debug/rotation_eval

    # Limit to a few folders first to check quickly
    python eval_rotation_model.py \\
        --raw-root  /path/to/2026Data \\
        --labels    /path/to/crop_labels.json \\
        --model     /path/to/models/rotation_v3.pth \\
        --out-dir   /path/to/debug/rotation_eval \\
        --limit     10 \\
        --samples   10

OUTPUT
------
    rotation_eval/
        panels/
            001_1965.jpg       ← visual grid for each folder
            002_1966.jpg
            ...
        summary.csv            ← folder, avg_angle, std_deg, n_kept, n_dropped
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Model — must match train_rotation_model_v3_cpu.py exactly
# ---------------------------------------------------------------------------

class RotationModel(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)
        in_features = base.classifier[0].in_features
        base.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        self.net = base

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_PREPROCESS = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

DEVICE = torch.device("cpu")


def load_model(model_path: Path) -> RotationModel:
    model = RotationModel()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def predict_angle(model: RotationModel, img_bgr: np.ndarray) -> float:
    """Returns predicted rotation angle in degrees [0, 360)."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    t = _PREPROCESS(pil).unsqueeze(0)
    with torch.no_grad():
        out = model(t)[0]
    angle_rad = math.atan2(out[0].item(), out[1].item())
    return math.degrees(angle_rad) % 360.0


# ---------------------------------------------------------------------------
# Cropping — two-step to match training data exactly
#
# Training data pipeline (build_rotation_dataset_v2.py):
#   1. Source images are already pre-cropped coins on white backgrounds
#   2. crop_to_circle() runs contour detection to re-center and mask the coin
#
# Inference pipeline must match:
#   1. label_crop()       — cut the coin region from the raw image using (cx,cy,r)
#   2. recenter_to_circle() — contour re-center + mask, same as training step 2
# ---------------------------------------------------------------------------

def label_crop(img: np.ndarray, cx: float, cy: float, r: float,
               output_size: int = 512) -> np.ndarray:
    """
    Step 1: Cut a square region from the raw image using manual label coords.
    Applies a circular mask and resizes — produces a coin-on-white image
    that can be fed into recenter_to_circle().
    """
    h, w = img.shape[:2]
    side = max(1, int(math.ceil(r * 2)))
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))

    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x1 + side), min(h, y1 + side)
    if sx1 < sx2 and sy1 < sy2:
        dx1, dy1 = sx1 - x1, sy1 - y1
        canvas[dy1:dy1+(sy2-sy1), dx1:dx1+(sx2-sx1)] = img[sy1:sy2, sx1:sx2]

    mask = np.zeros((side, side), dtype=np.uint8)
    cv2.circle(mask, (side // 2, side // 2), max(1, int(round(r))), 255, -1)
    canvas[mask == 0] = 255

    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_CUBIC)


def recenter_to_circle(img_bgr: np.ndarray, output_size: int = 224) -> np.ndarray:
    """
    Step 2: Contour-detect the coin on the white background, re-center it,
    and apply a clean circular mask. Identical to crop_to_circle() in
    build_rotation_dataset_v2.py — this is what the model was trained on.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, non_white = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    non_white = cv2.morphologyEx(non_white, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        (cx, cy), radius = cv2.minEnclosingCircle(hull)
        cx, cy, radius = float(cx), float(cy), float(radius)
        if radius < min(h, w) * 0.1:
            cx, cy, radius = w / 2.0, h / 2.0, min(h, w) / 2.0 * 0.9
    else:
        cx, cy, radius = w / 2.0, h / 2.0, min(h, w) / 2.0 * 0.9

    radius_padded = radius * 1.04
    side = max(1, int(math.ceil(radius_padded * 2)))

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))

    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x1 + side), min(h, y1 + side)
    if sx1 < sx2 and sy1 < sy2:
        dx1, dy1 = sx1 - x1, sy1 - y1
        canvas[dy1:dy1+(sy2-sy1), dx1:dx1+(sx2-sx1)] = img_bgr[sy1:sy2, sx1:sx2]

    mask = np.zeros((side, side), dtype=np.uint8)
    cv2.circle(mask, (side // 2, side // 2), max(1, int(round(radius_padded))), 255, -1)
    canvas[mask == 0] = 255

    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_CUBIC)


def crop_coin(img: np.ndarray, cx: float, cy: float, r: float,
              output_size: int = 512) -> np.ndarray:
    """
    Full two-step crop matching the training pipeline exactly.
    Use this everywhere — eval, preprocessing, and inference.
    """
    coarse = label_crop(img, cx, cy, r, output_size=output_size)
    return recenter_to_circle(coarse, output_size=output_size)


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    rotated = pil.rotate(angle_deg, resample=Image.BICUBIC,
                         expand=False, fillcolor=(255, 255, 255))
    return cv2.cvtColor(np.array(rotated), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Circular statistics
# ---------------------------------------------------------------------------

def circular_mean_deg(angles: List[float]) -> float:
    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0


def filtered_circular_mean(
    angles: List[float], threshold_deg: float = 15.0
) -> Tuple[float, List[float], int]:
    if not angles:
        return 0.0, [], 0
    prelim = circular_mean_deg(angles)
    kept = [a for a in angles
            if abs((a - prelim + 180) % 360 - 180) <= threshold_deg]
    if not kept:
        kept = angles
    return circular_mean_deg(kept), kept, len(angles) - len(kept)


def angle_std_deg(angles: List[float], mean_deg: float) -> float:
    if len(angles) < 2:
        return 0.0
    diffs = [(a - mean_deg + 180) % 360 - 180 for a in angles]
    return math.sqrt(sum(d*d for d in diffs) / (len(diffs) - 1))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_angle_text(img: np.ndarray, text: str,
                    color=(0, 0, 0)) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def make_panel(folder_name: str,
               crops: List[np.ndarray],
               angles: List[float],
               avg_angle: float,
               std_deg: float,
               n_dropped: int,
               panel_size: int = 256) -> np.ndarray:
    """
    Build a visual panel:
      Row 1: up to 5 raw crops (thumbnails)
      Row 2: same crops rotated by -avg_angle
      Bottom bar: folder name, avg angle, std
    """
    n = min(len(crops), 5)
    crops = crops[:n]
    angles = angles[:n]
    thumb = panel_size // n

    row_raw = []
    row_rot = []
    for crop, angle in zip(crops, angles):
        c = cv2.resize(crop, (thumb, thumb))
        rotated = cv2.resize(rotate_image(crop, -avg_angle), (thumb, thumb))

        # Annotate individual predicted angle on raw
        c = draw_angle_text(c, f"{angle:.1f}", color=(0, 100, 255))
        row_raw.append(c)
        row_rot.append(rotated)

    raw_row  = np.concatenate(row_raw, axis=1)
    rot_row  = np.concatenate(row_rot, axis=1)

    # Info bar
    bar_h = 40
    bar = np.full((bar_h, raw_row.shape[1], 3), 40, dtype=np.uint8)
    info = (f"{folder_name}   avg={avg_angle:.1f}   "
            f"std={std_deg:.1f}   dropped={n_dropped}")
    cv2.putText(bar, info, (6, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Divider labels
    label_w = 60
    def label_row(row_img, text):
        lbl = np.full((row_img.shape[0], label_w, 3), 60, dtype=np.uint8)
        cv2.putText(lbl, text, (4, row_img.shape[0]//2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return np.concatenate([lbl, row_img], axis=1)

    raw_row = label_row(raw_row, "CROP")
    rot_row = label_row(rot_row, "FIXED")
    bar     = np.concatenate([np.full((bar_h, label_w, 3), 30, dtype=np.uint8), bar], axis=1)

    return np.concatenate([raw_row, rot_row, bar], axis=0)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def sample_evenly(images: List[Path], n: int) -> List[Path]:
    if n <= 0 or n >= len(images):
        return images
    step = len(images) / n
    return [images[int(i * step)] for i in range(n)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate rotation model on 2026Data.")
    p.add_argument("--raw-root",          type=Path, required=True)
    p.add_argument("--labels",            type=Path, required=True)
    p.add_argument("--model",             type=Path, required=True)
    p.add_argument("--out-dir",           type=Path, required=True)
    p.add_argument("--samples",           type=int,  default=15,
                   help="Images to sample per folder for angle estimation")
    p.add_argument("--outlier-threshold", type=float, default=15.0)
    p.add_argument("--limit",             type=int,  default=0,
                   help="Max folders to evaluate (0 = all)")
    p.add_argument("--crop-size",         type=int,  default=256,
                   help="Internal crop size before rotation (not saved)")
    p.add_argument("--panel-thumb-cols",  type=int,  default=5)
    return p.parse_args()


def main():
    args = parse_args()

    panels_dir = args.out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    with open(args.labels) as f:
        labels = json.load(f)

    print(f"Loading model from {args.model} …")
    model = load_model(args.model)
    print(f"Model loaded. Evaluating {len(labels)} folders.\n")

    summary_rows = []
    folder_names = sorted(labels.keys())
    if args.limit:
        folder_names = folder_names[:args.limit]

    for idx, folder_name in enumerate(folder_names, 1):
        folder = args.raw_root / folder_name
        if not folder.exists():
            print(f"  [skip] missing: {folder_name}")
            continue

        images = list_images(folder)
        if not images:
            print(f"  [skip] no images: {folder_name}")
            continue

        params   = labels[folder_name]
        cx, cy, r = float(params["cx"]), float(params["cy"]), float(params["r"])

        sample_paths = sample_evenly(images, args.samples)
        raw_angles: List[float] = []
        crops: List[np.ndarray] = []

        for img_path in sample_paths:
            raw = cv2.imread(str(img_path))
            if raw is None:
                continue
            crop = crop_coin(raw, cx, cy, r, output_size=args.crop_size)
            angle = predict_angle(model, crop)
            raw_angles.append(angle)
            crops.append(crop)

        if not raw_angles:
            print(f"  [warn] no readable images: {folder_name}")
            continue

        avg_angle, kept, n_dropped = filtered_circular_mean(
            raw_angles, args.outlier_threshold
        )
        std = angle_std_deg(kept, avg_angle)
        correction = -avg_angle

        # Flag uncertain folders (high std or many dropped)
        uncertain = std > 20.0 or n_dropped > len(raw_angles) * 0.3
        flag = " ⚠ HIGH UNCERTAINTY" if uncertain else ""

        print(
            f"[{idx:>3}/{len(folder_names)}] {folder_name:<20}  "
            f"avg={avg_angle:>7.2f}°  std={std:>5.2f}°  "
            f"kept={len(kept)}/{len(raw_angles)}  "
            f"correction={correction:>7.2f}°{flag}"
        )

        # Build and save panel
        panel = make_panel(
            folder_name, crops, raw_angles,
            avg_angle, std, n_dropped,
            panel_size=args.crop_size * args.panel_thumb_cols,
        )
        panel_path = panels_dir / f"{idx:03d}_{folder_name}.jpg"
        cv2.imwrite(str(panel_path), panel)

        summary_rows.append({
            "folder":     folder_name,
            "n_images":   len(images),
            "n_samples":  len(raw_angles),
            "n_kept":     len(kept),
            "n_dropped":  n_dropped,
            "avg_angle":  round(avg_angle, 2),
            "correction": round(correction, 2),
            "std_deg":    round(std, 2),
            "uncertain":  uncertain,
            "raw_angles": " | ".join(f"{a:.1f}" for a in raw_angles),
        })

    # Save summary CSV
    csv_path = args.out_dir / "summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    # Print uncertain folders at the end for easy review
    uncertain_rows = [r for r in summary_rows if r["uncertain"]]
    print(f"\n{'='*60}")
    print(f"Evaluated {len(summary_rows)} folders.")
    print(f"Panels saved to: {panels_dir}")
    print(f"Summary CSV:     {csv_path}")
    if uncertain_rows:
        print(f"\n⚠  {len(uncertain_rows)} folders with high uncertainty:")
        for r in uncertain_rows:
            print(f"   {r['folder']:<20}  std={r['std_deg']}°  "
                  f"dropped={r['n_dropped']}/{r['n_samples']}")
    else:
        print("\n✓ All folders look consistent.")


if __name__ == "__main__":
    main()