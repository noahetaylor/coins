"""
preprocess_dataset_v2.py
========================
Crops and rotation-corrects raw coin images using manual crop labels
and a trained rotation model.

IMPROVEMENTS OVER v1
---------------------
1. Outlier-filtered circular mean — removes predictions that deviate more
   than `--outlier-threshold` degrees from the preliminary mean before
   computing the final average.  Prevents one bad frame from skewing the
   whole folder.
2. Configurable `--rotation-samples` (default 15, up from 7) for a tighter
   angle estimate per folder.
3. Per-folder confidence report: shows raw angle predictions, std-dev, and
   how many samples were kept after outlier rejection.
4. `--dry-run` mode — prints what would be done without writing any files.
5. `--folders` filter — process only specific folder names (useful for
   re-running a subset).
6. Progress bar via tqdm (falls back gracefully if not installed).
7. All tunable constants exposed as CLI arguments.

USAGE
-----
    # Normal run
    python preprocess_dataset_v2.py

    # Dry run (no output written)
    python preprocess_dataset_v2.py --dry-run

    # Only reprocess two folders
    python preprocess_dataset_v2.py --folders 1965 1966

    # More samples, tighter outlier window
    python preprocess_dataset_v2.py --rotation-samples 20 --outlier-threshold 10
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

IMPORTANT_DIR = Path(__file__).resolve().parents[1]
if str(IMPORTANT_DIR) not in sys.path:
    sys.path.insert(0, str(IMPORTANT_DIR))

from preprocessing import (
    Circle,
    FINE_ROTATION_MODEL_PATH,
    LABELS_PATH,
    RAW_ROOT,
    circular_mean_deg,
    draw_circle_overlay,
    list_images,
    load_rotation_model,
    predict_rotation_angle,
    recenter_to_circle,
    rotate_image,
    sample_evenly,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Defaults (all overridable via CLI)
# ---------------------------------------------------------------------------

OUTPUT_ROOT       = RAW_ROOT.parent / "RotatedDatasetV2"
DEBUG_ROOT        = IMPORTANT_DIR / "debug" / "preprocess_v2"
OUTPUT_SIZE       = 512
ROTATION_SAMPLES  = 15          # up from 7
OUTLIER_THRESHOLD = 15.0        # degrees; predictions outside this are dropped
RADIUS_PADDING    = 0.0
CIRCULAR_MASK     = True
SAVE_DEBUG        = True
DEBUG_FOLDER_LIMIT = 20
BACKGROUND        = (255, 255, 255)


# ---------------------------------------------------------------------------
# Crop utility (unchanged from v1, kept local so this file is self-contained)
# ---------------------------------------------------------------------------

def crop_label_square(
    img_cv: np.ndarray,
    circle: Circle,
    output_size: int,
    radius_padding: float,
    circular_mask: bool,
    background: tuple,
) -> np.ndarray:
    h, w = img_cv.shape[:2]
    radius = float(circle.r) * (1.0 + radius_padding)
    side = max(1, int(math.ceil(radius * 2.0)))

    x1 = int(round(circle.cx - side / 2.0))
    y1 = int(round(circle.cy - side / 2.0))
    x2, y2 = x1 + side, y1 + side

    canvas = np.full((side, side, 3), background, dtype=np.uint8)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    if sx1 < sx2 and sy1 < sy2:
        dx1, dy1 = sx1 - x1, sy1 - y1
        canvas[dy1:dy1+(sy2-sy1), dx1:dx1+(sx2-sx1)] = img_cv[sy1:sy2, sx1:sx2]

    if circular_mask:
        mask = np.zeros((side, side), dtype=np.uint8)
        cv2.circle(mask, (side // 2, side // 2), max(1, int(round(radius))), 255, -1)
        canvas[mask == 0] = background

    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_CUBIC)


def mask_centered_coin(img_cv: np.ndarray, background: tuple) -> np.ndarray:
    h, w = img_cv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)
    out = img_cv.copy()
    out[mask == 0] = background
    return out


# ---------------------------------------------------------------------------
# Outlier-filtered circular mean
# ---------------------------------------------------------------------------

def filtered_circular_mean(angles: List[float], threshold_deg: float) -> tuple:
    """
    1. Compute a preliminary circular mean.
    2. Drop any angle that deviates > threshold_deg from that mean.
    3. Recompute mean on the kept angles.

    Returns (final_mean_deg, kept_angles, dropped_count).
    """
    if not angles:
        return 0.0, [], 0

    prelim = circular_mean_deg(angles)

    kept = []
    for a in angles:
        # Angular difference wrapped to [-180, 180]
        diff = (a - prelim + 180) % 360 - 180
        if abs(diff) <= threshold_deg:
            kept.append(a)

    if not kept:
        # All outliers? Fall back to full set (shouldn't happen normally)
        kept = angles

    final_mean = circular_mean_deg(kept)
    dropped = len(angles) - len(kept)
    return final_mean, kept, dropped


def angle_std_deg(angles: List[float], mean_deg: float) -> float:
    """Circular standard deviation in degrees."""
    if len(angles) < 2:
        return 0.0
    diffs = [((a - mean_deg + 180) % 360 - 180) for a in angles]
    variance = sum(d * d for d in diffs) / (len(diffs) - 1)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Per-folder processing
# ---------------------------------------------------------------------------

def process_folder(
    folder_name: str,
    circle: Circle,
    rotation_model,
    args,
) -> Optional[float]:
    """
    Returns the correction angle applied, or None on failure.
    """
    folder = args.raw_root / folder_name
    if not folder.exists():
        print(f"  [warn] missing folder: {folder_name}")
        return None

    images = list_images(folder)
    if not images:
        print(f"  [warn] no images: {folder_name}")
        return None

    # --- Rotation estimation ---
    sample_paths = sample_evenly(images, args.rotation_samples)
    raw_angles = []
    sample_crops = []

    for img_path in sample_paths:
        raw = cv2.imread(str(img_path))
        if raw is None:
            continue
        cropped = crop_label_square(
            raw, circle, args.output_size, args.radius_padding,
            args.circular_mask, args.background,
        )
        recentered = recenter_to_circle(cropped)
        angle = predict_rotation_angle(rotation_model, recentered)
        raw_angles.append(angle)
        sample_crops.append((img_path, raw, cropped))

    if not raw_angles:
        print(f"  [warn] no valid samples: {folder_name}")
        return None

    avg_angle, kept_angles, n_dropped = filtered_circular_mean(
        raw_angles, args.outlier_threshold
    )
    std = angle_std_deg(kept_angles, avg_angle)
    correction = -avg_angle

    print(
        f"  [{folder_name}] "
        f"imgs={len(images)}  samples={len(raw_angles)}  "
        f"kept={len(kept_angles)}  dropped={n_dropped}  "
        f"avg={avg_angle:.2f}°  std={std:.2f}°  "
        f"correction={correction:.2f}°"
    )

    if args.dry_run:
        return correction

    # --- Write output ---
    out_dir = args.output_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    iter_images = images
    if HAS_TQDM and len(images) > 20:
        iter_images = tqdm(images, desc=f"  {folder_name}", leave=False)

    for img_path in iter_images:
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"  [warn] could not read: {img_path}")
            continue
        cropped = crop_label_square(
            raw, circle, args.output_size, args.radius_padding,
            args.circular_mask, args.background,
        )
        rotated = rotate_image(cropped, correction)
        if args.circular_mask:
            rotated = mask_centered_coin(rotated, args.background)
        cv2.imwrite(str(out_dir / img_path.name), rotated)

    # --- Debug output ---
    if args.save_debug and sample_crops:
        DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
        sample_path, raw_img, cropped = sample_crops[len(sample_crops) // 2]
        rotated = rotate_image(cropped, correction)
        if args.circular_mask:
            rotated = mask_centered_coin(rotated, args.background)

        prefix = DEBUG_ROOT / f"{folder_name}_{sample_path.stem}"
        cv2.imwrite(str(prefix) + "_01_label_overlay.jpg",
                    draw_circle_overlay(raw_img, circle))
        cv2.imwrite(str(prefix) + "_02_cropped.jpg", cropped)
        cv2.imwrite(str(prefix) + "_03_rotated.jpg", rotated)

        info = (
            f"folder={folder_name}\n"
            f"sample={sample_path.name}\n"
            f"raw_size={raw_img.shape[1]}x{raw_img.shape[0]}\n"
            f"label_cx={circle.cx:.2f}\n"
            f"label_cy={circle.cy:.2f}\n"
            f"label_r={circle.r:.2f}\n"
            f"raw_angles={[f'{a:.2f}' for a in raw_angles]}\n"
            f"kept_angles={[f'{a:.2f}' for a in kept_angles]}\n"
            f"n_dropped={n_dropped}\n"
            f"avg_angle={avg_angle:.2f}\n"
            f"std_deg={std:.2f}\n"
            f"correction={correction:.2f}\n"
        )
        (DEBUG_ROOT / f"{folder_name}_{sample_path.stem}_info.txt").write_text(info)

    return correction


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess coin dataset v2.")
    p.add_argument("--raw-root",          type=Path,  default=RAW_ROOT)
    p.add_argument("--labels",            type=Path,  default=LABELS_PATH)
    p.add_argument("--rotation-model",    type=Path,  default=FINE_ROTATION_MODEL_PATH)
    p.add_argument("--output-root",       type=Path,  default=OUTPUT_ROOT)
    p.add_argument("--output-size",       type=int,   default=OUTPUT_SIZE)
    p.add_argument("--rotation-samples",  type=int,   default=ROTATION_SAMPLES)
    p.add_argument("--outlier-threshold", type=float, default=OUTLIER_THRESHOLD,
                   help="Max degrees from mean to keep a sample (default 15)")
    p.add_argument("--radius-padding",    type=float, default=RADIUS_PADDING)
    p.add_argument("--no-circular-mask",  action="store_true")
    p.add_argument("--save-debug",        action="store_true", default=SAVE_DEBUG)
    p.add_argument("--debug-limit",       type=int,   default=DEBUG_FOLDER_LIMIT)
    p.add_argument("--dry-run",           action="store_true",
                   help="Print what would be done without writing any files")
    p.add_argument("--folders",           nargs="*",  default=None,
                   help="Only process these folder names (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    args.circular_mask = not args.no_circular_mask
    args.background    = BACKGROUND

    if args.dry_run:
        print("*** DRY RUN — no files will be written ***\n")

    print(f"Labels         : {args.labels}")
    print(f"Rotation model : {args.rotation_model}")
    print(f"Output root    : {args.output_root}")
    print(f"Samples/folder : {args.rotation_samples}")
    print(f"Outlier thresh : {args.outlier_threshold}°")
    print(f"Circular mask  : {args.circular_mask}")
    print()

    with open(args.labels) as f:
        labels = json.load(f)

    if args.folders:
        labels = {k: v for k, v in labels.items() if k in args.folders}
        print(f"Filtering to {len(labels)} requested folders.")

    rotation_model = load_rotation_model(args.rotation_model)

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)

    corrections = {}
    for idx, (folder_name, params) in enumerate(labels.items(), start=1):
        save_debug = (
            args.save_debug and
            (not args.debug_limit or idx <= args.debug_limit) and
            not args.dry_run
        )
        args.save_debug_this = save_debug
        correction = process_folder(
            folder_name,
            Circle.from_mapping(params),
            rotation_model,
            args,
        )
        if correction is not None:
            corrections[folder_name] = correction

    print(f"\nDone. Processed {len(corrections)}/{len(labels)} folders.")
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()