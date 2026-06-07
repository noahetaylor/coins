"""
build_crop_dataset.py
=====================
Generates a synthetic training dataset for the coin crop model.

WHY THIS EXISTS
---------------
The raw 2026Data images all have the coin at nearly the same position
(cx ~2290, cy ~1625, r ~1200) because the camera rig is fixed.  A model
trained directly on those images has nothing spatial to learn — it just
memorizes the mean and collapses.

STRATEGY
---------
1. Use crop_labels.json to extract each coin tightly from its raw image.
2. Re-paste the cropped coin at a *random* position and scale on a
   synthetic background (solid colour, noise, or a real non-coin image).
3. Record the new (cx, cy, r) in normalised coordinates [0, 1].
4. Save every composite as a JPEG + a single labels JSON.

The result is a dataset where the coin appears all over the frame at
many sizes — exactly the variation the model needs.

USAGE
-----
    python build_crop_dataset.py \
        --raw-root   /path/to/2026Data \
        --labels     /path/to/crop_labels.json \
        --out-dir    /path/to/CropDatasetSynthetic \
        --per-folder 60 \
        --output-size 512 \
        --backgrounds /optional/folder/of/bg/images

OUTPUT STRUCTURE
----------------
    CropDatasetSynthetic/
        images/          ← composite JPEG files
        labels.json      ← {filename: {cx, cy, r, w, h}} (pixel coords)
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {s.lower() for s in IMAGE_SUFFIXES})


def extract_coin(raw: np.ndarray, cx: float, cy: float, r: float, padding: float = 0.05) -> tuple:
    """
    Cut a square region containing the coin from raw.
    Returns (coin_img, mask) where mask is a uint8 circle mask (255=coin).
    """
    h, w = raw.shape[:2]
    r_pad = r * (1.0 + padding)
    side = max(1, int(math.ceil(r_pad * 2)))

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = x1 + side
    y2 = y1 + side

    # Canvas with white fill for out-of-bounds regions
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    if sx1 < sx2 and sy1 < sy2:
        dx1 = sx1 - x1
        dy1 = sy1 - y1
        canvas[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = raw[sy1:sy2, sx1:sx2]

    # Circular mask
    mask = np.zeros((side, side), dtype=np.uint8)
    cv2.circle(mask, (side // 2, side // 2), max(1, int(round(r_pad))), 255, -1)

    return canvas, mask


def make_background(height: int, width: int, bg_images: list) -> np.ndarray:
    """Generate or sample a background image."""
    if bg_images and random.random() < 0.6:
        # Use a real background image
        bg_path = random.choice(bg_images)
        bg = cv2.imread(str(bg_path))
        if bg is not None:
            bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_LINEAR)
            return bg

    choice = random.random()
    if choice < 0.4:
        # Solid colour
        colour = [random.randint(30, 220) for _ in range(3)]
        return np.full((height, width, 3), colour, dtype=np.uint8)
    elif choice < 0.7:
        # Gaussian noise
        noise = np.random.randint(30, 220, (height, width, 3), dtype=np.uint8)
        # Smooth it slightly so it looks like a real surface
        noise = cv2.GaussianBlur(noise, (15, 15), 0)
        return noise
    else:
        # Gradient
        top = np.array([random.randint(30, 200)] * 3, dtype=np.float32)
        bot = np.array([random.randint(30, 200)] * 3, dtype=np.float32)
        grad = np.zeros((height, width, 3), dtype=np.uint8)
        for row in range(height):
            t = row / max(height - 1, 1)
            grad[row] = np.clip(top * (1 - t) + bot * t, 0, 255).astype(np.uint8)
        return grad


def paste_coin(
    bg: np.ndarray,
    coin: np.ndarray,
    mask: np.ndarray,
    cx_frac: float,
    cy_frac: float,
    scale: float,
) -> tuple:
    """
    Resize coin to `scale` fraction of bg min-dim and paste centred at
    (cx_frac, cy_frac) in normalised bg coordinates.

    Returns (composite, cx_px, cy_px, r_px) — the actual circle in pixel
    coords on the composite.
    """
    H, W = bg.shape[:2]
    coin_side_orig = coin.shape[0]  # square
    r_orig = coin_side_orig / 2.0

    target_r = scale * min(H, W) / 2.0
    target_side = max(1, int(round(target_r * 2)))

    coin_rs = cv2.resize(coin, (target_side, target_side), interpolation=cv2.INTER_CUBIC)
    mask_rs = cv2.resize(mask, (target_side, target_side), interpolation=cv2.INTER_NEAREST)

    cx_px = cx_frac * W
    cy_px = cy_frac * H

    x1 = int(round(cx_px - target_side / 2))
    y1 = int(round(cy_px - target_side / 2))
    x2 = x1 + target_side
    y2 = y1 + target_side

    composite = bg.copy()

    # Clamp to canvas
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = target_side - max(0, x2 - W)
    src_y2 = target_side - max(0, y2 - H)
    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x1 < src_x2 and src_y1 < src_y2:
        roi_mask = mask_rs[src_y1:src_y2, src_x1:src_x2]
        roi_coin = coin_rs[src_y1:src_y2, src_x1:src_x2]
        roi_bg = composite[dst_y1:dst_y2, dst_x1:dst_x2]

        m = roi_mask[:, :, np.newaxis] / 255.0
        composite[dst_y1:dst_y2, dst_x1:dst_x2] = (
            roi_coin * m + roi_bg * (1.0 - m)
        ).astype(np.uint8)

    # Actual circle centre/radius in the full composite
    true_cx = cx_px
    true_cy = cy_px
    true_r = target_r

    return composite, true_cx, true_cy, true_r


def add_augmentation(img: np.ndarray) -> np.ndarray:
    """Apply mild photometric augmentation."""
    # Brightness / contrast jitter
    alpha = random.uniform(0.75, 1.3)   # contrast
    beta = random.randint(-25, 25)       # brightness
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Slight blur sometimes
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build synthetic crop training dataset.")
    p.add_argument("--raw-root", type=Path, required=True,
                   help="Root of raw coin folders (2026Data)")
    p.add_argument("--labels", type=Path, required=True,
                   help="Path to crop_labels.json")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory")
    p.add_argument("--per-folder", type=int, default=60,
                   help="Synthetic composites to generate per source folder")
    p.add_argument("--output-size", type=int, default=512,
                   help="Output image size (square)")
    p.add_argument("--backgrounds", type=Path, default=None,
                   help="Optional folder of background images to sample from")
    p.add_argument("--min-scale", type=float, default=0.30,
                   help="Min coin radius as fraction of canvas min-dim")
    p.add_argument("--max-scale", type=float, default=0.75,
                   help="Max coin radius as fraction of canvas min-dim")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="Fraction of composites to put in val/ split")
    p.add_argument("--source-samples", type=int, default=5,
                   help="How many raw images to sample per folder as coin sources")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.labels) as f:
        labels = json.load(f)

    bg_images = []
    if args.backgrounds and args.backgrounds.is_dir():
        bg_images = list_images(args.backgrounds)
        print(f"Loaded {len(bg_images)} background images from {args.backgrounds}")

    out_train = args.out_dir / "train" / "images"
    out_val = args.out_dir / "val" / "images"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    all_labels = {"train": {}, "val": {}}
    S = args.output_size
    created = 0
    skipped = 0

    folder_names = sorted(labels.keys())
    print(f"Processing {len(folder_names)} folders, {args.per_folder} composites each …")

    for folder_name in folder_names:
        folder = args.raw_root / folder_name
        if not folder.exists():
            print(f"  [warn] missing folder: {folder_name}")
            skipped += 1
            continue

        images = list_images(folder)
        if not images:
            print(f"  [warn] no images in: {folder_name}")
            skipped += 1
            continue

        params = labels[folder_name]
        cx_lbl = float(params["cx"])
        cy_lbl = float(params["cy"])
        r_lbl = float(params["r"])

        # Sample a few raw images as coin sources for this folder
        sources = random.sample(images, min(args.source_samples, len(images)))

        # Pre-extract coins from those source images
        coins = []
        for img_path in sources:
            raw = cv2.imread(str(img_path))
            if raw is None:
                continue
            coin, mask = extract_coin(raw, cx_lbl, cy_lbl, r_lbl)
            coins.append((coin, mask))

        if not coins:
            print(f"  [warn] could not read any images: {folder_name}")
            skipped += 1
            continue

        folder_label_train = {}
        folder_label_val = {}

        for i in range(args.per_folder):
            coin, mask = random.choice(coins)

            # Random placement — allow slight overflow so model sees edge coins
            scale = random.uniform(args.min_scale, args.max_scale)

            # Ensure full circle stays inside the frame
            min_frac = scale
            max_frac = 1.0 - scale

            cx_frac = random.uniform(min_frac, max_frac)
            cy_frac = random.uniform(min_frac, max_frac)


            bg = make_background(S, S, bg_images)
            composite, true_cx, true_cy, true_r = paste_coin(
                bg, coin, mask, cx_frac, cy_frac, scale
            )
            composite = add_augmentation(composite)

            fname = f"{folder_name}_{i:04d}.jpg"
            is_val = (random.random() < args.val_fraction)
            split = "val" if is_val else "train"
            out_path = (out_val if is_val else out_train) / fname
            cv2.imwrite(str(out_path), composite)

            entry = {
                "cx": true_cx,
                "cy": true_cy,
                "r": true_r,
                "w": S,
                "h": S,
                "folder": folder_name,
            }
            all_labels[split][fname] = entry
            created += 1

        print(f"  [{folder_name}] → {args.per_folder} composites")

    # Save labels
    for split in ("train", "val"):
        lbl_path = args.out_dir / split / "labels.json"
        with open(lbl_path, "w") as f:
            json.dump(all_labels[split], f, indent=2)
        print(f"Saved {len(all_labels[split])} {split} labels → {lbl_path}")

    print(f"\nDone. Created {created} composites, skipped {skipped} folders.")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()