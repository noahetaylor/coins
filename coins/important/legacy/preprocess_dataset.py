import json
import math
import sys
from pathlib import Path

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
    rotate_image,
    sample_evenly,
)


# VS Code play-button defaults.
OUTPUT_ROOT = RAW_ROOT.parent / "RotatedDatasetManualV2"
DEBUG_ROOT = IMPORTANT_DIR / "debug" / "preprocess_v2"
OUTPUT_SIZE = 512
ROTATION_SAMPLES = 7
ROTATION_CORRECTION_SIGN = -1.0
RADIUS_PADDING = 0.0
CIRCULAR_MASK = True
SAVE_DEBUG = True
DEBUG_FOLDER_LIMIT = 20
BACKGROUND = (255, 255, 255)


def crop_label_square(
    img_cv: np.ndarray,
    circle: Circle,
    output_size: int = OUTPUT_SIZE,
    radius_padding: float = RADIUS_PADDING,
    circular_mask: bool = CIRCULAR_MASK,
) -> np.ndarray:
    """Crop exactly around a labeled coin circle, padding outside the raw image with white."""
    h, w = img_cv.shape[:2]
    radius = float(circle.r) * (1.0 + radius_padding)
    side = max(1, int(math.ceil(radius * 2.0)))

    x1 = int(round(circle.cx - side / 2.0))
    y1 = int(round(circle.cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side

    canvas = np.full((side, side, 3), BACKGROUND, dtype=np.uint8)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)
    if src_x1 < src_x2 and src_y1 < src_y2:
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        canvas[dst_y1:dst_y1 + (src_y2 - src_y1), dst_x1:dst_x1 + (src_x2 - src_x1)] = img_cv[src_y1:src_y2, src_x1:src_x2]

    if circular_mask:
        mask = np.zeros((side, side), dtype=np.uint8)
        center = (side // 2, side // 2)
        cv2.circle(mask, center, max(1, int(round(radius))), 255, -1)
        canvas[mask == 0] = BACKGROUND

    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_CUBIC)


def mask_centered_coin(img_cv: np.ndarray) -> np.ndarray:
    if not CIRCULAR_MASK:
        return img_cv
    h, w = img_cv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)
    out = img_cv.copy()
    out[mask == 0] = BACKGROUND
    return out


def save_debug_images(folder_name: str, sample_path: Path, raw, cropped, rotated, circle: Circle, avg_angle: float) -> None:
    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    prefix = DEBUG_ROOT / f"{folder_name}_{sample_path.stem}"
    cv2.imwrite(str(prefix) + "_01_label_overlay.jpg", draw_circle_overlay(raw, circle))
    cv2.imwrite(str(prefix) + "_02_cropped.jpg", cropped)
    cv2.imwrite(str(prefix) + "_03_rotated.jpg", rotated)

    h, w = raw.shape[:2]
    info = (
        f"folder={folder_name}\n"
        f"sample={sample_path.name}\n"
        f"raw_size={w}x{h}\n"
        f"label_cx={circle.cx:.2f}\n"
        f"label_cy={circle.cy:.2f}\n"
        f"label_r={circle.r:.2f}\n"
        f"avg_angle={avg_angle:.2f}\n"
        f"correction={ROTATION_CORRECTION_SIGN * avg_angle:.2f}\n"
    )
    (DEBUG_ROOT / f"{folder_name}_{sample_path.stem}_info.txt").write_text(info)


def process_folder(folder_name: str, circle: Circle, rotation_model, save_debug: bool) -> None:
    folder = RAW_ROOT / folder_name
    if not folder.exists():
        print(f"[warn] missing folder: {folder_name}")
        return

    images = list_images(folder)
    if not images:
        print(f"[warn] no images in folder: {folder_name}")
        return

    sample_paths = sample_evenly(images, ROTATION_SAMPLES)
    angles = []
    sample_crops = []
    for img_path in sample_paths:
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"[warn] could not read sample: {img_path}")
            continue
        cropped = crop_label_square(raw, circle)
        angles.append(predict_rotation_angle(rotation_model, cropped))
        sample_crops.append((img_path, raw, cropped))

    if not angles:
        print(f"[warn] no valid rotation samples: {folder_name}")
        return

    avg_angle = circular_mean_deg(angles)
    correction = ROTATION_CORRECTION_SIGN * avg_angle
    print(f"[{folder_name}] images={len(images)} samples={len(angles)} avg={avg_angle:.2f} correction={correction:.2f}")

    out_dir = OUTPUT_ROOT / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"[warn] could not read image: {img_path}")
            continue
        cropped = crop_label_square(raw, circle)
        rotated = rotate_image(cropped, correction)
        rotated = mask_centered_coin(rotated)
        cv2.imwrite(str(out_dir / img_path.name), rotated)

    if save_debug and sample_crops:
        sample_path, raw, cropped = sample_crops[len(sample_crops) // 2]
        rotated = mask_centered_coin(rotate_image(cropped, correction))
        save_debug_images(folder_name, sample_path, raw, cropped, rotated, circle, avg_angle)


def main():
    print("Manual preprocessing v2")
    print(f"labels: {LABELS_PATH}")
    print(f"rotation model: {FINE_ROTATION_MODEL_PATH}")
    print(f"output: {OUTPUT_ROOT}")
    print(f"debug: {DEBUG_ROOT}")

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    rotation_model = load_rotation_model(FINE_ROTATION_MODEL_PATH)

    for idx, (folder_name, params) in enumerate(labels.items(), start=1):
        save_debug = SAVE_DEBUG and (not DEBUG_FOLDER_LIMIT or idx <= DEBUG_FOLDER_LIMIT)
        process_folder(folder_name, Circle.from_mapping(params), rotation_model, save_debug)

    print("Done.")


if __name__ == "__main__":
    main()
