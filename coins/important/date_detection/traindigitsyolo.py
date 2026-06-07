import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO digit detector for coin date OCR.")
    parser.add_argument("--data", default="data/coins/digitsannotations/dataset2.yml", help="Path to dataset YAML file for digit training.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Initial weights for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=256, help="Image size for training.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--device", default="cuda" if Path("/usr/bin/nvidia-smi").exists() else "cpu", help="Training device name or index.")
    parser.add_argument("--save-path", default="data/coins/Bounds/Heads/models/digit_detector.pt", help="Path to save the trained digit model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.train(
        data=str(Path(args.data)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=True,
    )
    model.save(str(Path(args.save_path)))
    print(f"Saved trained digit detector to {args.save_path}")


if __name__ == "__main__":
    main()
