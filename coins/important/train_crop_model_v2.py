"""
train_crop_model_v2.py
======================
Trains a coin crop-circle regression model on the *synthetic* dataset
produced by build_crop_dataset.py.

KEY IMPROVEMENTS OVER v1
-------------------------
1. Reads the synthetic dataset (coins at varied positions/scales) instead
   of the nearly-constant real-image positions.
2. Spatial augmentation: random horizontal/vertical flip, small translate,
   slight zoom — all applied consistently to image AND label.
3. Dual loss: SmoothL1 on (cx, cy, r) + soft-IoU on the predicted circle,
   which is differentiable and directly penalises bad circle overlap.
4. EfficientNet-B0 backbone option (better accuracy for same parameter count).
5. LR scheduler (cosine annealing) instead of fixed LR.
6. Per-epoch pixel-MAE breakdown (cx error, cy error, r error) for easier
   debugging.
7. Saves both the best-val checkpoint AND the final checkpoint.

USAGE
-----
    python train_crop_model_v2.py \
        --data-dir /path/to/CropDatasetSynthetic \
        --out      /path/to/models/crop_model_v2.pth \
        --backbone resnet18 \
        --epochs   80

ARGS
----
--data-dir     Root of the synthetic dataset (must contain train/ and val/)
--out          Where to save the best model weights
--backbone     resnet18 | resnet34 | efficientnet_b0  (default: resnet18)
--epochs       Training epochs (default: 80)
--batch-size   (default: 32)
--lr           Initial learning rate (default: 3e-4)
--weight-decay (default: 1e-4)
--seed         (default: 42)
--num-workers  DataLoader workers (default: 4)
--no-iou-loss  Disable soft-IoU term (use pure SmoothL1 only)
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticCropDataset(Dataset):
    """
    Loads composites from build_crop_dataset.py output.

    Each item returns:
        img_tensor  : (3, 224, 224) float normalised
        target      : (3,) float — [cx/W, cy/H, r/max(W,H)] in [0, 1]
        dims        : (2,) float — [W, H] in original output_size pixels
    """

    def __init__(self, images_dir: Path, labels_path: Path, augment: bool):
        self.images_dir = images_dir
        self.augment = augment

        with open(labels_path) as f:
            raw = json.load(f)

        self.samples = []
        for fname, info in raw.items():
            self.samples.append((
                fname,
                float(info["cx"]),
                float(info["cy"]),
                float(info["r"]),
                float(info["w"]),
                float(info["h"]),
            ))

        print(f"  {len(self.samples)} samples from {labels_path}")

    def __len__(self):
        return len(self.samples)

    def _spatial_augment(self, img_pil, cx, cy, r, W, H):
        """
        Apply random horizontal / vertical flip and small translate.
        Returns updated (img_pil, cx, cy, r).
        """
        # Horizontal flip
        if random.random() < 0.5:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            cx = W - cx

        # Vertical flip
        if random.random() < 0.5:
            img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
            cy = H - cy

        # Small translate (up to ±8% of image size)
        if random.random() < 0.5:
            dx = random.uniform(-0.08, 0.08) * W
            dy = random.uniform(-0.08, 0.08) * H
            img_np = np.array(img_pil)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            W_i = int(W)
            H_i = int(H)
            img_np = cv2.warpAffine(
                img_np, M, (W_i, H_i),
                borderMode=cv2.BORDER_REFLECT_101,
            )

            img_pil = Image.fromarray(img_np)
            cx = np.clip(cx + dx, 0, W)
            cy = np.clip(cy + dy, 0, H)

        return img_pil, cx, cy, r

    # Normalisation constants (ImageNet)
    _norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    _to_tensor = transforms.ToTensor()
    _resize = transforms.Resize((224, 224))

    def __getitem__(self, idx):
        fname, cx, cy, r, W, H = self.samples[idx]
        img_path = self.images_dir / fname

        img = Image.open(img_path).convert("RGB")

        if self.augment:
            # Photometric
            img = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            )(img)
            if random.random() < 0.15:
                img = transforms.RandomAutocontrast(p=1.0)(img)

            # Spatial (label-aware)
            img, cx, cy, r = self._spatial_augment(img, cx, cy, r, W, H)

        img = self._resize(img)
        img_t = self._norm(self._to_tensor(img))

        target = torch.tensor([
            cx / W,
            cy / H,
            r / max(W, H),
        ], dtype=torch.float32).clamp(0.0, 1.0)

        dims = torch.tensor([W, H], dtype=torch.float32)

        return img_t, target, dims


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(backbone_name: str) -> nn.Module:
    if backbone_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 3)
    elif backbone_name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 3)
    elif backbone_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 3),
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # Wrap so forward always returns sigmoid-clamped output
    class _Wrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return torch.sigmoid(self.base(x))

    return _Wrapper(m)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    """
    Differentiable approximate IoU loss for circles.
    Works in pixel space for numerical stability.
    Both pred and target are (B, 3) with [cx/W, cy/H, r/maxD].
    dims is (B, 2) with [W, H].
    """
    W = dims[:, 0]
    H = dims[:, 1]
    maxD = torch.maximum(W, H)

    def to_px(t):
        cx = t[:, 0] * W
        cy = t[:, 1] * H
        r  = t[:, 2] * maxD
        return cx, cy, r

    pcx, pcy, pr = to_px(pred)
    tcx, tcy, tr = to_px(target)

    # Distance between centres
    d = torch.sqrt((pcx - tcx) ** 2 + (pcy - tcy) ** 2 + 1e-6)

    # Intersection area (approximate via min-enclosing formula)
    # Use the lens-area formula for two circles.
    # Clamped to [0, union] for stability.
    r1, r2 = pr.clamp(min=1.0), tr.clamp(min=1.0)
    intersection = torch.zeros_like(d)

    # Case: one circle inside the other
    inside = d <= (r1 - r2).abs()
    r_small = torch.minimum(r1, r2)
    intersection = torch.where(inside, math.pi * r_small ** 2, intersection)

    # Case: partial overlap
    overlap = (~inside) & (d < r1 + r2)
    if overlap.any():
        d_ov = d[overlap]
        r1_ov = r1[overlap]
        r2_ov = r2[overlap]
        cos_a1 = ((d_ov**2 + r1_ov**2 - r2_ov**2) / (2 * d_ov * r1_ov)).clamp(-1, 1)
        cos_a2 = ((d_ov**2 + r2_ov**2 - r1_ov**2) / (2 * d_ov * r2_ov)).clamp(-1, 1)
        a1 = torch.acos(cos_a1)
        a2 = torch.acos(cos_a2)
        area = (r1_ov**2 * (a1 - torch.sin(a1)*torch.cos(a1)) +
                r2_ov**2 * (a2 - torch.sin(a2)*torch.cos(a2)))
        intersection[overlap] = area

    union = math.pi * r1**2 + math.pi * r2**2 - intersection + 1e-6
    iou = intersection / union
    return (1.0 - iou).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pixel_errors(pred, target, dims):
    """Returns mean absolute pixel errors for (cx, cy, r)."""
    W, H = dims[:, 0], dims[:, 1]
    maxD = torch.maximum(W, H)
    scale = torch.stack([W, H, maxD], dim=1)
    err = torch.abs((pred - target) * scale)
    return err.mean(dim=0).cpu().tolist()  # [cx_err, cy_err, r_err]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, use_iou_loss, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    cx_e = cy_e = r_e = 0.0
    n = 0

    for imgs, targets, dims in loader:
        imgs    = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        dims    = dims.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            preds = model(imgs)
            loss = criterion(preds, targets)
            if use_iou_loss:
                loss = loss + 0.5 * soft_iou_loss(preds, targets, dims)
            if is_train:
                loss.backward()
                optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        errs = pixel_errors(preds.detach(), targets, dims)
        cx_e += errs[0] * bs
        cy_e += errs[1] * bs
        r_e  += errs[2] * bs
        n += bs

    return total_loss / n, cx_e / n, cy_e / n, r_e / n


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18", "resnet34", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-iou-loss", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dir   = args.data_dir / "train"
    val_dir     = args.data_dir / "val"
    train_imgs  = train_dir / "images"
    val_imgs    = val_dir   / "images"
    train_lbls  = train_dir / "labels.json"
    val_lbls    = val_dir   / "labels.json"

    for p in [train_imgs, val_imgs, train_lbls, val_lbls]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    print("Loading datasets …")
    train_ds = SyntheticCropDataset(train_imgs, train_lbls, augment=True)
    val_ds   = SyntheticCropDataset(val_imgs,   val_lbls,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(DEVICE == "cuda"),
    )

    print(f"Building {args.backbone} model …")
    model = build_model(args.backbone).to(DEVICE)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    use_iou = not args.no_iou_loss

    print(f"\nTraining for {args.epochs} epochs  |  IoU loss: {use_iou}\n")
    print(f"{'Ep':>4}  {'TLoss':>8}  {'VLoss':>8}  {'cx_err':>7}  {'cy_err':>7}  {'r_err':>7}  LR")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        t_loss, _, _, _ = run_epoch(model, train_loader, criterion, use_iou, optimizer)
        v_loss, cx_e, cy_e, r_e = run_epoch(model, val_loader, criterion, use_iou)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:>4}  {t_loss:>8.5f}  {v_loss:>8.5f}  "
            f"{cx_e:>7.1f}  {cy_e:>7.1f}  {r_e:>7.1f}  {lr_now:.2e}"
        )

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), args.out)
            print(f"       ↳ saved new best → {args.out}")

    # Also save final
    final_path = args.out.with_stem(args.out.stem + "_final")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Best val loss: {best_val:.5f}")
    print(f"Best model : {args.out}")
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()