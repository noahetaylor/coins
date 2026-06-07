import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMPORTANT_DIR = Path(__file__).resolve().parents[1]
if str(IMPORTANT_DIR) not in sys.path:
    sys.path.insert(0, str(IMPORTANT_DIR))

from preprocessing import CROP_MODEL_PATH, LABELS_PATH, RAW_ROOT, IMAGE_SUFFIXES


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
    transforms.RandomAutocontrast(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


class CropDataset(Dataset):
    def __init__(self, root: Path, labels: dict, folder_names, samples_per_folder: int, transform):
        self.samples = []
        self.transform = transform

        for folder_name in folder_names:
            folder = root / folder_name
            if not folder.exists():
                print(f"[warn] Skipping missing folder: {folder_name}")
                continue

            images = list_images(folder)
            if not images:
                print(f"[warn] No images in {folder_name}")
                continue

            if samples_per_folder > 0 and len(images) > samples_per_folder:
                images = random.sample(images, samples_per_folder)

            params = labels[folder_name]
            for img_path in images:
                self.samples.append((img_path, float(params["cx"]), float(params["cy"]), float(params["r"])))

        print(f"Loaded {len(self.samples)} samples from {len(folder_names)} folders")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cx_raw, cy_raw, r_raw = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        target = torch.tensor([
            cx_raw / w,
            cy_raw / h,
            r_raw / max(w, h),
        ], dtype=torch.float32)
        target = torch.clamp(target, 0.0, 1.0)

        return self.transform(img), target, torch.tensor([w, h], dtype=torch.float32)


class BoundedCropModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 3)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))


def split_folders(folder_names, val_fraction: float, seed: int):
    folders = list(folder_names)
    rng = random.Random(seed)
    rng.shuffle(folders)
    val_count = max(1, int(round(len(folders) * val_fraction))) if len(folders) > 1 else 0
    return folders[val_count:], folders[:val_count]


def pixel_mae(outputs, targets, dims):
    scale = torch.stack([dims[:, 0], dims[:, 1], torch.maximum(dims[:, 0], dims[:, 1])], dim=1)
    return torch.mean(torch.abs((outputs - targets) * scale)).item()


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_pixel_mae = 0.0
    total = 0

    for imgs, targets, dims in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        dims = dims.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        total_pixel_mae += pixel_mae(outputs.detach(), targets, dims) * batch_size
        total += batch_size

    return total_loss / total, total_pixel_mae / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train the coin crop circle regression model.")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--out", type=Path, default=CROP_MODEL_PATH)
    parser.add_argument("--samples-per-folder", type=int, default=0, help="0 means use every image.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.labels, "r") as f:
        labels = json.load(f)

    existing_folders = [name for name in labels if (args.raw_root / name).exists()]
    train_folders, val_folders = split_folders(existing_folders, args.val_fraction, args.seed)
    print(f"Train folders: {len(train_folders)} | Val folders: {len(val_folders)}")

    train_ds = CropDataset(args.raw_root, labels, train_folders, args.samples_per_folder, train_transform)
    val_ds = CropDataset(args.raw_root, labels, val_folders, args.samples_per_folder, eval_transform)
    if not train_ds or not val_ds:
        raise RuntimeError("Need at least one training and validation sample.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=DEVICE == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=DEVICE == "cuda",
    )

    model = BoundedCropModel().to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_px = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_px = run_epoch(model, val_loader, criterion)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train={train_loss:.6f} ({train_px:.1f}px) | "
            f"val={val_loss:.6f} ({val_px:.1f}px)"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.backbone.state_dict(), args.out)
            print(f"Saved new best model -> {args.out}")

    print("Training complete.")


if __name__ == "__main__":
    main()
