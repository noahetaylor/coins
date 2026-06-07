import argparse
import json
import sys
from pathlib import Path

import cv2

IMPORTANT_DIR = Path(__file__).resolve().parents[1]
if str(IMPORTANT_DIR) not in sys.path:
    sys.path.insert(0, str(IMPORTANT_DIR))

from preprocessing import (
    Circle,
    LABELS_PATH,
    RAW_ROOT,
    crop_to_coin,
    draw_circle_overlay,
    list_images,
    load_crop_model,
    load_rotation_model,
    predict_crop_circle,
    predict_rotation_angle,
    rotate_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Save crop-model prediction debug images.")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("coins/important/debug/crop_rotate"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--mask", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.labels, "r") as f:
        labels = json.load(f)

    crop_model = load_crop_model()
    rotation_model = load_rotation_model()

    written = 0
    for folder_name, params in labels.items():
        folder = args.raw_root / folder_name
        if not folder.exists():
            continue
        images = list_images(folder)
        if not images:
            continue

        img_path = images[len(images) // 2]
        raw = cv2.imread(str(img_path))
        if raw is None:
            continue

        manual_circle = Circle.from_mapping(params)
        predicted_circle = predict_crop_circle(crop_model, raw)
        cropped = crop_to_coin(raw, predicted_circle, output_size=args.output_size, padding=args.padding, circular_mask=args.mask)
        angle = predict_rotation_angle(rotation_model, cropped)
        upright = rotate_image(cropped, -angle)

        prefix = args.output_dir / f"{written:03d}_{folder_name}_{img_path.stem}"
        cv2.imwrite(str(prefix) + "_manual_circle.jpg", draw_circle_overlay(raw, manual_circle, color=(255, 0, 0)))
        cv2.imwrite(str(prefix) + "_pred_circle.jpg", draw_circle_overlay(raw, predicted_circle, color=(0, 0, 255)))
        cv2.imwrite(str(prefix) + "_pred_crop.jpg", cropped)
        cv2.imwrite(str(prefix) + "_upright.jpg", upright)
        print(
            f"{folder_name}: manual=({manual_circle.cx:.1f},{manual_circle.cy:.1f},{manual_circle.r:.1f}) "
            f"pred=({predicted_circle.cx:.1f},{predicted_circle.cy:.1f},{predicted_circle.r:.1f}) "
            f"angle={angle:.2f}"
        )

        written += 1
        if written >= args.limit:
            break

    print(f"Saved {written} debug sets to {args.output_dir}")


if __name__ == "__main__":
    main()
