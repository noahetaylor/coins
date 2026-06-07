import argparse
from pathlib import Path
from typing import Optional
import cv2


def convert_yolo_to_pixels(box: str, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    parts = box.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO box specification: {box}")

    class_id, x_center, y_center, width, height = map(float, parts[:5])
    x_center = int(x_center * img_w)
    y_center = int(y_center * img_h)
    box_width = int(width * img_w)
    box_height = int(height * img_h)

    x1 = max(0, x_center - box_width // 2)
    y1 = max(0, y_center - box_height // 2)
    x2 = min(img_w, x_center + box_width // 2)
    y2 = min(img_h, y_center + box_height // 2)
    return x1, y1, x2, y2


def crop_labeled_images(
    image_root: Path,
    output_dir: Path,
    class_id: Optional[int] = None,
    recursive: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(image_root.rglob("*.jpg")) if recursive else sorted(image_root.glob("*.jpg"))

    for image_path in image_paths:
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"WARNING: Unable to read image: {image_path}")
            continue

        with label_path.open("r", encoding="utf-8") as handle:
            boxes = [line.strip() for line in handle if line.strip()]

        for idx, box in enumerate(boxes):
            parts = box.split()
            if len(parts) < 5:
                continue
            if class_id is not None and int(float(parts[0])) != class_id:
                continue

            x1, y1, x2, y2 = convert_yolo_to_pixels(box, image.shape[1], image.shape[0])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            relative_name = image_path.relative_to(image_root).with_suffix("")
            safe_name = "_".join(relative_name.parts)
            out_name = f"{safe_name}_crop_{idx}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), crop)
            print(f"Saved crop: {out_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract cropped date regions from YOLO-labeled coin images.")
    parser.add_argument("--image-root", default="data/coins/Bounds/Heads/Testing", help="Directory containing labeled images.")
    parser.add_argument("--output-dir", default="data/coins/date_crops2", help="Directory to save cropped regions.")
    parser.add_argument("--class-id", type=int, default=None, help="Optional YOLO class id to filter crops.")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive search for images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crop_labeled_images(
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        class_id=args.class_id,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
