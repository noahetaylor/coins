import argparse
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

DEFAULT_BASE_DIR = Path("data/coins/Bounds/Heads")
DEFAULT_DATE_DATASET = "dataset.yml"
DEFAULT_DIGIT_DATASET = "dataset_digits.yml"
DEFAULT_MODEL_DIR = "models"
DEFAULT_DATE_MODEL_NAME = "date_detector.pt"
DEFAULT_DIGIT_MODEL_NAME = "digit_detector.pt"
DEFAULT_VAL_DIR = "Testing"
DEFAULT_VALID_YEARS = {str(y) for y in range(1965, 1984)}
DEFAULT_DATE_CONF = 0.25
DEFAULT_DIGIT_CONF = 0.4
DEFAULT_PAD = 0.15


def configure_logging() -> None:
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.INFO,
    )


def build_paths(
    base_dir: Path,
    model_dir: Optional[Path] = None,
    date_dataset: Optional[Path] = None,
    digit_dataset: Optional[Path] = None,
    val_dir: Optional[Path] = None,
) -> dict:
    base_dir = Path(base_dir)
    model_dir = Path(model_dir or base_dir / DEFAULT_MODEL_DIR)
    date_dataset = Path(date_dataset or base_dir / DEFAULT_DATE_DATASET)
    digit_dataset = Path(digit_dataset or base_dir / DEFAULT_DIGIT_DATASET)
    val_dir = Path(val_dir or base_dir / DEFAULT_VAL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "model_dir": model_dir,
        "date_dataset": date_dataset,
        "digit_dataset": digit_dataset,
        "date_model": model_dir / DEFAULT_DATE_MODEL_NAME,
        "digit_model": model_dir / DEFAULT_DIGIT_MODEL_NAME,
        "val_dir": val_dir,
    }


def collect_images(folder: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def train_detector(
    data_path: Path,
    model_path: Path,
    weights: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    amp: bool,
) -> YOLO:
    logging.info("Training YOLO model: %s", model_path.name)
    model = YOLO(weights)
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        amp=amp,
        verbose=True,
    )
    model.save(str(model_path))
    logging.info("Saved model to %s", model_path)
    return model


def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logging.info("Loading model from %s", model_path)
    return YOLO(str(model_path))


def detect_date_region(
    image_path: Path,
    model: YOLO,
    conf: float = DEFAULT_DATE_CONF,
    pad: float = DEFAULT_PAD,
    show: bool = False,
) -> Optional[object]:
    image = cv2.imread(str(image_path))
    if image is None:
        logging.warning("Unable to read image: %s", image_path)
        return None

    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    best_box = max(boxes, key=lambda b: float(b.conf[0]) if hasattr(b, "conf") else 0.0)
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

    h, w = image.shape[:2]
    pw = int((x2 - x1) * pad)
    ph = int((y2 - y1) * pad)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    if show:
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.title("Cropped Date Region")
        plt.axis("off")
        plt.show()

    return crop


def extract_year(
    crop: object,
    digit_model: YOLO,
    valid_years: set,
    conf: float = DEFAULT_DIGIT_CONF,
    show: bool = False,
) -> str:
    if crop is None:
        return "UNKNOWN"

    results = digit_model.predict(source=crop, conf=conf, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return "UNKNOWN"

    digits = []
    overlay = crop.copy()

    for box in results[0].boxes:
        cls_value = int(box.cls[0]) if hasattr(box, "cls") else None
        if cls_value is None or not 0 <= cls_value <= 9:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        digits.append((x1, str(cls_value)))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            str(cls_value),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    if not digits:
        return "UNKNOWN"

    digits.sort(key=lambda x: x[0])
    year = "".join([value for _, value in digits])
    if year not in valid_years:
        return "UNKNOWN"

    if show:
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"Digit Detection: {year}")
        plt.axis("off")
        plt.show()

    return year


def get_ground_truth_from_folder(img_path: Path, valid_years: set) -> Optional[str]:
    folder = img_path.parent.name
    return folder if folder in valid_years else None


def test_samples(
    folder: Path,
    date_model: YOLO,
    digit_model: YOLO,
    valid_years: set,
    n: int = 5,
    randomize: bool = False,
    conf_date: float = DEFAULT_DATE_CONF,
    conf_digit: float = DEFAULT_DIGIT_CONF,
) -> None:
    images = collect_images(folder)
    if not images:
        logging.warning("No images found in %s", folder)
        return

    if randomize:
        import random

        random.shuffle(images)

    for img_path in images[:n]:
        crop = detect_date_region(img_path, date_model, conf=conf_date)
        year = extract_year(crop, digit_model, valid_years, conf=conf_digit)
        logging.info("%s -> %s", img_path, year)


def evaluate_model(
    folder: Path,
    date_model: YOLO,
    digit_model: YOLO,
    valid_years: set,
    conf_date: float = DEFAULT_DATE_CONF,
    conf_digit: float = DEFAULT_DIGIT_CONF,
    report_csv: Optional[Path] = None,
) -> dict:
    images = collect_images(folder)
    total = 0
    skipped = 0
    correct = 0
    detected = 0
    recognized = 0
    failures = []

    for img_path in images:
        truth = get_ground_truth_from_folder(img_path, valid_years)
        if truth is None:
            skipped += 1
            continue

        total += 1
        crop = detect_date_region(img_path, date_model, conf=conf_date)
        if crop is None:
            failures.append((img_path, truth, "NO_DATE"))
            continue

        detected += 1
        prediction = extract_year(crop, digit_model, valid_years, conf=conf_digit)
        if prediction != "UNKNOWN":
            recognized += 1
        if prediction == truth:
            correct += 1
        else:
            failures.append((img_path, truth, prediction))

    accuracy = correct / total if total else 0.0
    detection_rate = detected / total if total else 0.0
    recognition_rate = recognized / total if total else 0.0
    unknown_rate = sum(1 for _, _, pred in failures if pred == "UNKNOWN") / total if total else 0.0

    logging.info("Evaluation results for %s", folder)
    logging.info("Total valid images: %d", total)
    logging.info("Skipped invalid folders: %d", skipped)
    logging.info("Correct predictions: %d", correct)
    logging.info("Date region detected: %.2f%%", detection_rate * 100)
    logging.info("Recognized year: %.2f%%", recognition_rate * 100)
    logging.info("Overall accuracy: %.2f%%", accuracy * 100)
    logging.info("Unknown predictions: %.2f%%", unknown_rate * 100)

    if report_csv is not None:
        import csv

        report_csv.parent.mkdir(parents=True, exist_ok=True)
        with report_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "ground_truth", "prediction"])
            for image_path, truth, pred in failures:
                writer.writerow([str(image_path), truth, pred])
        logging.info("Saved failure report to %s", report_csv)

    return {
        "total": total,
        "skipped": skipped,
        "correct": correct,
        "detected": detected,
        "recognized": recognized,
        "accuracy": accuracy,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COIN DATE IDENTIFICATION PIPELINE using Ultralytics YOLO"
    )
    parser.add_argument(
        "--mode",
        choices=["train-date", "train-digit", "train-all", "eval", "test"],
        default="eval",
        help="Execution mode for the pipeline.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Root folder for date dataset and model output.",
    )
    parser.add_argument(
        "--date-dataset",
        default=None,
        help="Path to the YOLO dataset YAML for date detection.",
    )
    parser.add_argument(
        "--digit-dataset",
        default=None,
        help="Path to the YOLO dataset YAML for digit detection.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory to save or load trained models.",
    )
    parser.add_argument(
        "--val-dir",
        default=None,
        help="Root validation folder containing dated subfolders.",
    )
    parser.add_argument(
        "--date-weights",
        default="yolov8n.pt",
        help="Pretrained weights for date-region training.",
    )
    parser.add_argument(
        "--digit-weights",
        default="yolov8n.pt",
        help="Pretrained weights for digit training.",
    )
    parser.add_argument("--epochs-date", type=int, default=30, help="Date detector epochs.")
    parser.add_argument("--epochs-digit", type=int, default=40, help="Digit detector epochs.")
    parser.add_argument("--imgsz", type=int, default=256, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size.")
    parser.add_argument("--device", default="0", help="Training device: 0, 1, or cpu.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training.")
    parser.add_argument("--date-conf", type=float, default=DEFAULT_DATE_CONF, help="Detection confidence for date region.")
    parser.add_argument("--digit-conf", type=float, default=DEFAULT_DIGIT_CONF, help="Detection confidence for digits.")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample images for test mode.")
    parser.add_argument("--randomize", action="store_true", help="Randomize sample order for test mode.")
    parser.add_argument("--failure-csv", default=None, help="Optional CSV path for failure report.")
    parser.add_argument("--valid-years", nargs="*", default=None, help="List of valid year strings for evaluation.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    paths = build_paths(
        Path(args.base_dir),
        model_dir=Path(args.model_dir) if args.model_dir else None,
        date_dataset=Path(args.date_dataset) if args.date_dataset else None,
        digit_dataset=Path(args.digit_dataset) if args.digit_dataset else None,
        val_dir=Path(args.val_dir) if args.val_dir else None,
    )

    valid_years = set(args.valid_years) if args.valid_years else DEFAULT_VALID_YEARS

    if args.mode in {"train-date", "train-all"}:
        train_detector(
            data_path=paths["date_dataset"],
            model_path=paths["date_model"],
            weights=args.date_weights,
            epochs=args.epochs_date,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            amp=args.amp,
        )

    if args.mode in {"train-digit", "train-all"}:
        train_detector(
            data_path=paths["digit_dataset"],
            model_path=paths["digit_model"],
            weights=args.digit_weights,
            epochs=args.epochs_digit,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            amp=args.amp,
        )

    if args.mode in {"eval", "test"}:
        date_model = load_model(paths["date_model"])
        digit_model = load_model(paths["digit_model"])

        if args.mode == "test":
            test_samples(
                folder=paths["val_dir"],
                date_model=date_model,
                digit_model=digit_model,
                valid_years=valid_years,
                n=args.samples,
                randomize=args.randomize,
                conf_date=args.date_conf,
                conf_digit=args.digit_conf,
            )
        else:
            report_path = Path(args.failure_csv) if args.failure_csv else None
            evaluate_model(
                folder=paths["val_dir"],
                date_model=date_model,
                digit_model=digit_model,
                valid_years=valid_years,
                conf_date=args.date_conf,
                conf_digit=args.digit_conf,
                report_csv=report_path,
            )


if __name__ == "__main__":
    main()
