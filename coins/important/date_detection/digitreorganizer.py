import argparse
import random
from pathlib import Path
import shutil

DEFAULT_DIGIT_SRC = Path("data/coins/digitsannotations/date_cropmatchedout222")
DEFAULT_HEADS_BASE = Path("data/coins/Bounds/Heads")
DEFAULT_DIGIT_BASE = DEFAULT_HEADS_BASE / "digits"
DEFAULT_TRAIN_IMG = DEFAULT_DIGIT_BASE / "train/images"
DEFAULT_TRAIN_LBL = DEFAULT_DIGIT_BASE / "train/labels"
DEFAULT_VAL_IMG = DEFAULT_DIGIT_BASE / "val/images"
DEFAULT_VAL_LBL = DEFAULT_DIGIT_BASE / "val/labels"


def quietly_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image_list(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Digit source directory does not exist: {source_dir}")
    return sorted(
        [
            p
            for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def split_and_copy(
    files: list[Path],
    train_dir: Path,
    val_dir: Path,
    val_ratio: float,
    seed: int,
) -> None:
    random.seed(seed)
    random.shuffle(files)
    split = int(len(files) * (1.0 - val_ratio))
    train_files = files[:split]
    val_files = files[split:]

    for dest_dir in [train_dir / "images", train_dir / "labels", val_dir / "images", val_dir / "labels"]:
        quietly_mkdir(dest_dir)

    print(f"Total digit images: {len(files)}")
    print(f"Training set: {len(train_files)}")
    print(f"Validation set: {len(val_files)}")

    for src_files, dest_img_dir, dest_lbl_dir in [
        (train_files, train_dir / "images", train_dir / "labels"),
        (val_files, val_dir / "images", val_dir / "labels"),
    ]:
        for image_path in src_files:
            label_path = image_path.with_suffix(".txt")
            if not label_path.exists():
                print(f"WARNING: Missing label for {image_path}")
                continue
            shutil.copy2(image_path, dest_img_dir / image_path.name)
            shutil.copy2(label_path, dest_lbl_dir / label_path.name)

    print("Digit dataset conversion complete!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLO digit dataset from labeled digit crops.")
    parser.add_argument("--source", default=str(DEFAULT_DIGIT_SRC), help="Source folder containing digit image crops and labels.")
    parser.add_argument("--dest-base", default=str(DEFAULT_DIGIT_BASE), help="Destination root for YOLO digit dataset.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/validation split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    dest_base = Path(args.dest_base)
    train_dir = dest_base / "train"
    val_dir = dest_base / "val"

    files = load_image_list(source)
    if not files:
        raise SystemExit(f"No digit images found in {source}")

    split_and_copy(files, train_dir, val_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
