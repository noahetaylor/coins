"""
FAST CPU Rotation Trainer
EfficientNet-Lite0 + 160px input + minimal augmentation
"""

import argparse
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import models, transforms

DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

ANGLE_RE = re.compile(r"_(\d+(?:\.\d+)?)(?:_a\d+)?$")


# ---------------------------------------------------------------------------
# Model: EfficientNet-Lite0 (CPU-optimized)
# ---------------------------------------------------------------------------

class RotationModel(nn.Module):
    """EfficientNet-Lite0 with sin/cos regression head."""
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        base = models.efficientnet_b0(weights=None)  # Lite0 uses same structure
        in_features = base.classifier[1].in_features

        base.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # sin, cos
        )

        self.net = base

    def forward(self, x):
        return self.net(x)

    def freeze_backbone(self):
        for name, p in self.net.named_parameters():
            if "classifier" not in name:
                p.requires_grad_(False)
        print("  Backbone frozen")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)
        print("  Backbone unfrozen")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def parse_angle(path: Path) -> float:
    m = ANGLE_RE.search(path.stem)
    if not m:
        raise ValueError(f"Cannot parse angle from: {path.name}")
    return float(m.group(1)) % 360.0


class RotationDataset(Dataset):
    def __init__(self, root: Path, augment: bool, input_size: int = 160):
        exts = {".jpg", ".jpeg", ".png"}
        self.files = sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        self.augment = augment

        self._base = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(f"  {len(self.files):,} images in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        angle_deg = parse_angle(path)

        rad = math.radians(angle_deg)
        target = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)

        img = Image.open(path).convert("RGB")
        return self._base(img), target


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mean_angle_error_deg(preds, targets):
    pred_angle   = torch.atan2(preds[:, 0], preds[:, 1])
    target_angle = torch.atan2(targets[:, 0], targets[:, 1])
    diff = torch.atan2(torch.sin(pred_angle - target_angle),
                       torch.cos(pred_angle - target_angle))
    return diff.abs().mean().item() * (180.0 / math.pi)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = total_mae = n = 0

    for imgs, targets in loader:
        if is_train:
            optimizer.zero_grad()

        preds = model(imgs)
        loss = criterion(preds, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_mae  += mean_angle_error_deg(preds.detach(), targets) * bs
        n += bs

    return total_loss / n, total_mae / n


def make_loader(ds, batch_size, num_workers, max_images, shuffle):
    if shuffle and max_images > 0 and max_images < len(ds):
        sampler = RandomSampler(ds, replacement=False, num_samples=max_images)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out",      type=Path, required=True)
    p.add_argument("--epochs",   type=int,  default=20)
    p.add_argument("--freeze-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-head",  type=float, default=1e-3)
    p.add_argument("--lr-full",  type=float, default=3e-5)
    p.add_argument("--max-train-images", type=int, default=5000)
    p.add_argument("--input-size", type=int, default=160)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    train_dir = args.data_dir / "train"
    val_dir   = args.data_dir / "val"

    print("\nLoading datasets…")
    train_ds = RotationDataset(train_dir, augment=False, input_size=args.input_size)
    val_ds   = RotationDataset(val_dir,   augment=False, input_size=args.input_size)

    val_loader = make_loader(val_ds, args.batch_size, args.num_workers,
                             max_images=0, shuffle=False)

    model = RotationModel().to(DEVICE)
    criterion = nn.MSELoss()

    best_val = float("inf")

    print("\nStarting training…\n")

    for epoch in range(1, args.epochs + 1):

        if epoch == 1:
            model.freeze_backbone()
            optimizer = optim.Adam(model.net.classifier.parameters(), lr=args.lr_head)
            phase = "head"

        elif epoch == args.freeze_epochs + 1:
            model.unfreeze_all()
            optimizer = optim.Adam(model.parameters(), lr=args.lr_full)
            phase = "full"

        train_loader = make_loader(train_ds, args.batch_size, args.num_workers,
                                   args.max_train_images, shuffle=True)

        t0 = time.time()
        t_loss, t_mae = run_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_mae = run_epoch(model, val_loader, criterion)
        dt = time.time() - t0

        print(f"{epoch:>3} | {phase:>4} | "
              f"T-MAE {t_mae:6.2f}° | V-MAE {v_mae:6.2f}° | {dt:5.1f}s")

        if v_mae < best_val:
            best_val = v_mae
            torch.save(model.state_dict(), args.out)
            print(f"   ↳ Saved new best model ({v_mae:.2f}°)")

    print("\nDone.")
    print(f"Best val MAE: {best_val:.2f}°")


if __name__ == "__main__":
    main()
