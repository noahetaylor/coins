import argparse
import shutil
from pathlib import Path

DEFAULT_BASE = Path("data/coins/Bounds/Heads")
DEFAULT_TRAIN_SRC = DEFAULT_BASE / "Training"
DEFAULT_VAL_SRC = DEFAULT_BASE / "Testing"
DEFAULT_IMAGES_TRAIN = DEFAULT_BASE / "images/train"
DEFAULT_IMAGES_VAL = DEFAULT_BASE / "images/val"
DEFAULT_LABELS_TRAIN = DEFAULT_BASE / "labels/train"
DEFAULT_LABELS_VAL = DEFAULT_BASE / "labels/val"


def quietly_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_year_folder(src_root: Path, dest_images: Path, dest_labels: Path) -> None:
    if not src_root.exists():
        raise FileNotFoundError(f"Source folder does not exist: {src_root}")

    quietly_mkdir(dest_images)
    quietly_mkdir(dest_labels)

    for year_dir in sorted(src_root.iterdir()):
        if not year_dir.is_dir():
            continue

        for image_path in sorted(year_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = image_path.with_suffix(".txt")
            if not label_path.exists():
                print(f"WARNING: Missing label for {image_path}")
                continue

            shutil.copy2(image_path, dest_images / image_path.name)
            shutil.copy2(label_path, dest_labels / label_path.name)

    print(f"Copied data from {src_root} to {dest_images}" )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset folders from year-labeled coin images.")
    parser.add_argument("--train-src", default=str(DEFAULT_TRAIN_SRC), help="Source directory for training year folders.")
    parser.add_argument("--val-src", default=str(DEFAULT_VAL_SRC), help="Source directory for validation year folders.")
    parser.add_argument("--images-train", default=str(DEFAULT_IMAGES_TRAIN), help="Destination train images directory.")
    parser.add_argument("--labels-train", default=str(DEFAULT_LABELS_TRAIN), help="Destination train labels directory.")
    parser.add_argument("--images-val", default=str(DEFAULT_IMAGES_VAL), help="Destination validation images directory.")
    parser.add_argument("--labels-val", default=str(DEFAULT_LABELS_VAL), help="Destination validation labels directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_year_folder(Path(args.train_src), Path(args.images_train), Path(args.labels_train))
    copy_year_folder(Path(args.val_src), Path(args.images_val), Path(args.labels_val))
    print("Dataset conversion complete!")


if __name__ == "__main__":
    main()
