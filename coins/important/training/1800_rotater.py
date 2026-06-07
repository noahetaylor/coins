import re
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMPORTANT_DIR = Path(__file__).resolve().parents[1]

TRAIN_DIR = Path("/home/noah/projects/image-identifier/data/coins/FineRotationDataset/train")
TEST_DIR  = Path("/home/noah/projects/image-identifier/data/coins/FineRotationDataset/test")

OUT_PATH = IMPORTANT_DIR / "models" / "fine_rotation_full.pth"

BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ============================================================
# DATASET
# ============================================================
ANGLE_RE = re.compile(r"_(\d+(?:\.\d+)?)$")

def parse_angle_from_name(path: Path) -> float:
    """
    Extract angle from filename like:
    1965_img_0023_123.00.jpg → 123.00
    """
    stem = path.stem
    m = ANGLE_RE.search(stem)
    if not m:
        raise ValueError(f"Cannot parse angle from filename: {path.name}")
    return float(m.group(1)) % 360.0

class RotationDataset(Dataset):
    def __init__(self, root: Path):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        self.images = [p for ext in exts for p in root.glob(ext)]
        self.images = sorted(self.images)

        print(f"Loaded {len(self.images)} images from {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        angle_deg = parse_angle_from_name(path)

        # Convert to sin/cos
        rad = math.radians(angle_deg)
        target = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)

        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        img_t = transform(pil)

        return img_t, target

# ============================================================
# MODEL
# ============================================================
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # sin, cos
    return model

# ============================================================
# TRAINING LOOP
# ============================================================
def main():
    torch.manual_seed(42)

    train_ds = RotationDataset(TRAIN_DIR)
    test_ds  = RotationDataset(TEST_DIR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # -------------------------
        # TRAIN
        # -------------------------
        model.train()
        total_loss = 0.0

        for imgs, targets in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        train_loss = total_loss / len(train_ds)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(test_ds)

        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUT_PATH)
            print(f"  Saved new best model → {OUT_PATH}")

    print("Training complete")

if __name__ == "__main__":
    main()
