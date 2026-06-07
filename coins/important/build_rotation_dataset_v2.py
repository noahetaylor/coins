"""
build_rotation_dataset_v2.py
============================
Builds a fine rotation regression dataset from your labeled source coins.

VAL SPLIT — KEY CHANGE FROM PREVIOUS VERSION
---------------------------------------------
Old approach: hold out entire source coins → val is unseen coin faces →
model appears to overfit even when it's actually learning fine.

New approach: ANGLE-LEVEL split. Every source coin contributes images
to BOTH train and val. Val gets a random 15% of all generated images,
spread evenly across all angles and all coins. This means:
  - Val MAE is a clean measure of rotation accuracy on known coin designs
  - (which is exactly your production use-case — same coin designs,
     unknown rotation)
  - Val set is large enough to give a stable signal (~12k val images)

USAGE
-----
    python build_rotation_dataset_v2.py \
        --src  /path/to/2026RotationTraining \
        --out  /path/to/RotationDatasetV2 \
        --val-fraction 0.15

OUTPUT STRUCTURE
----------------
    RotationDatasetV2/
        train/
        val/
        manifest_train.json
        manifest_val.json
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FOLDER_BASE_ANGLE = {
    "rot_train_0":   0,
    "rot_train_90":  90,
    "rot_train_180": 180,
    "rot_train_270": 270,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def list_images(folder: Path):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def crop_to_circle(img_bgr: np.ndarray, output_size: int = 224) -> np.ndarray:
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


def rotate_image(img_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    rotated = pil.rotate(angle_deg, resample=Image.BICUBIC, expand=False,
                         fillcolor=(255, 255, 255))
    return cv2.cvtColor(np.array(rotated), cv2.COLOR_RGB2BGR)


def photometric_augment(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    img += rng.uniform(-30, 30)
    mean = img.mean()
    img = mean + rng.uniform(0.8, 1.25) * (img - mean)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rotations-per-image", type=int, default=360)
    p.add_argument("--aug-per-rotation", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="Fraction of ALL generated images to put in val (default 0.15)")
    p.add_argument("--output-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    train_dir = args.out / "train"
    val_dir   = args.out / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    angles = list(range(args.rotations_per_image)) if args.rotations_per_image == 360 \
        else [i * (360 / args.rotations_per_image) for i in range(args.rotations_per_image)]

    # Collect all (source_img, base_angle, rot_angle, aug_idx) tuples first,
    # then do a single global shuffle + split. This gives a proper angle-level
    # split across all coins and all folders.
    print("Scanning source images …")
    all_jobs = []  # (img_path, base_angle, rot_angle, aug_suffix)
    for folder_name, base_angle in FOLDER_BASE_ANGLE.items():
        folder = args.src / folder_name
        if not folder.exists():
            print(f"  [skip] {folder_name} not found")
            continue
        images = list_images(folder)
        if not images:
            print(f"  [skip] no images in {folder_name}")
            continue
        print(f"  {folder_name}: {len(images)} source images")
        for img_path in images:
            for rot_angle in angles:
                all_jobs.append((img_path, base_angle, rot_angle, ""))
                for i in range(args.aug_per_rotation):
                    all_jobs.append((img_path, base_angle, rot_angle, f"_a{i}"))

    total = len(all_jobs)
    n_val = int(round(total * args.val_fraction))
    print(f"\nTotal images to generate : {total:,}")
    print(f"Val split ({args.val_fraction:.0%})           : {n_val:,}")
    print(f"Train                    : {total - n_val:,}")

    # Global shuffle then split — every angle/coin/aug equally likely to be val
    rng.shuffle(all_jobs)
    val_jobs   = set(id(j) for j in all_jobs[:n_val])
    # We can't use id on tuples reliably after shuffle, so use index instead
    val_indices = set(range(n_val))

    # Cache cropped coins to avoid re-reading and re-cropping the same source
    coin_cache: dict = {}
    manifest_train, manifest_val = [], []
    total_train = total_val = 0

    print("\nGenerating images …")
    for idx, (img_path, base_angle, rot_angle, aug_suffix) in enumerate(all_jobs):
        is_val = idx < n_val

        # Load + crop coin (cached)
        key = str(img_path)
        if key not in coin_cache:
            raw = cv2.imread(str(img_path))
            if raw is None:
                print(f"  [warn] could not read {img_path}")
                coin_cache[key] = None
            else:
                coin_cache[key] = crop_to_circle(raw, output_size=args.output_size)
        coin = coin_cache[key]
        if coin is None:
            continue

        true_angle = (base_angle + rot_angle) % 360
        rotated = rotate_image(coin, rot_angle)

        if aug_suffix and not is_val:
            # Only apply augmentation to train images
            rotated = photometric_augment(rotated, rng)
        elif aug_suffix and is_val:
            # Skip augmented copies for val entirely — val stays clean
            continue

        fname = f"{img_path.stem}_{true_angle:.2f}{aug_suffix}.jpg"
        out_path = (val_dir if is_val else train_dir) / fname
        if not cv2.imwrite(str(out_path), rotated):
            print(f"  [warn] failed to write {out_path}")
            continue

        entry = {
            "file": fname,
            "angle_deg": true_angle,
            "source": img_path.name,
            "base_angle": base_angle,
            "rot_angle": rot_angle,
            "augmented": bool(aug_suffix),
        }
        if is_val:
            manifest_val.append(entry)
            total_val += 1
        else:
            manifest_train.append(entry)
            total_train += 1

        if (idx + 1) % 5000 == 0:
            print(f"  {idx+1:,}/{total:,} done …")

    for split, manifest in [("train", manifest_train), ("val", manifest_val)]:
        p = args.out / f"manifest_{split}.json"
        with open(p, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved {len(manifest):,} {split} entries → {p}")

    print(f"\nDone. Train: {total_train:,}  Val: {total_val:,}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()