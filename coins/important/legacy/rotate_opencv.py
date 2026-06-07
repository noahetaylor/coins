import argparse
import sys
from pathlib import Path

import cv2

IMPORTANT_DIR = Path(__file__).resolve().parents[1]
if str(IMPORTANT_DIR) not in sys.path:
    sys.path.insert(0, str(IMPORTANT_DIR))

from preprocessing import (
    RAW_ROOT,
    ROTATED_ROOT,
    circular_mean_deg,
    crop_to_coin,
    list_images,
    load_crop_model,
    load_rotation_model,
    predict_crop_circle,
    predict_rotation_angle,
    rotate_image,
    sample_evenly,
)


def process_folder(folder_path: Path, crop_model, rotation_model, args) -> None:
    images = list_images(folder_path)
    if not images:
        return

    sample_paths = sample_evenly(images, args.samples)
    circles = []
    angles = []
    for img_path in sample_paths:
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"[warn] Could not read sample: {img_path}")
            continue
        circle = predict_crop_circle(crop_model, raw)
        circles.append(circle)
        cropped = crop_to_coin(raw, circle, output_size=args.output_size, padding=args.padding, circular_mask=args.mask)
        angles.append(predict_rotation_angle(rotation_model, cropped))

    if not circles or not angles:
        print(f"[warn] No readable samples in {folder_path.name}")
        return

    avg_circle = type(circles[0])(
        sum(c.cx for c in circles) / len(circles),
        sum(c.cy for c in circles) / len(circles),
        sum(c.r for c in circles) / len(circles),
    )
    avg_angle = circular_mean_deg(angles)
    print(
        f"[{folder_path.name}] samples={len(angles)} "
        f"crop=({avg_circle.cx:.1f},{avg_circle.cy:.1f},{avg_circle.r:.1f}) "
        f"avg_angle={avg_angle:.2f}"
    )

    rel = folder_path.relative_to(args.raw_root)
    out_dir = args.output_root / rel
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"[warn] Could not read image: {img_path}")
            continue
        cropped = crop_to_coin(raw, avg_circle, output_size=args.output_size, padding=args.padding, circular_mask=args.mask)
        upright = rotate_image(cropped, -avg_angle)
        if not args.dry_run:
            cv2.imwrite(str(out_dir / img_path.name), upright)


def parse_args():
    parser = argparse.ArgumentParser(description="Experimental ML-crop coin preprocessing pipeline.")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROTATED_ROOT)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-folders", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    crop_model = load_crop_model()
    rotation_model = load_rotation_model()

    folders = sorted(p for p in args.raw_root.iterdir() if p.is_dir())
    if args.limit_folders:
        folders = folders[:args.limit_folders]

    for folder in folders:
        process_folder(folder, crop_model, rotation_model, args)


if __name__ == "__main__":
    main()
