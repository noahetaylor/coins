"""
eval_crop_model_v2.py
=====================
Runs the v2 crop model against real raw images and saves visual debug grids
showing predicted vs. manual-label circles side by side.

USAGE
-----
    python eval_crop_model_v2.py \
        --raw-root  /path/to/2026Data \
        --labels    /path/to/crop_labels.json \
        --model     /path/to/models/crop_model_v2.pth \
        --out-dir   /path/to/debug/crop_eval \
        --backbone  resnet18 \
        --limit     20
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Rebuild the model architecture (must match train_crop_model_v2.py)
# ---------------------------------------------------------------------------

def build_model(backbone_name: str) -> nn.Module:
    if backbone_name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 3)
    elif backbone_name == "resnet34":
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 3)
    elif backbone_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 3))
    else:
        raise ValueError(backbone_name)

    class _Wrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return torch.sigmoid(self.base(x))

    return _Wrapper(m)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(model: nn.Module, img_bgr: np.ndarray):
    """Returns (cx, cy, r) in pixel coords."""
    h, w = img_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    t = _preprocess(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)[0].cpu().tolist()
    cx = out[0] * w
    cy = out[1] * h
    r  = out[2] * max(w, h)
    return cx, cy, r


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_circle_on(img: np.ndarray, cx: float, cy: float, r: float,
                   colour, label: str) -> np.ndarray:
    out = img.copy()
    cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(r))
    cv2.circle(out, (cx_i, cy_i), max(1, r_i), colour, 3)
    cv2.circle(out, (cx_i, cy_i), 6, colour, -1)
    cv2.putText(out, label, (cx_i + 10, cy_i - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 2, cv2.LINE_AA)
    return out


def side_by_side(left: np.ndarray, right: np.ndarray, target_h: int = 600) -> np.ndarray:
    def _resize(img):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (int(w * scale), target_h))
    l = _resize(left)
    r = _resize(right)
    return np.concatenate([l, r], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--labels",   type=Path, required=True)
    p.add_argument("--model",    type=Path, required=True)
    p.add_argument("--out-dir",  type=Path, required=True)
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18", "resnet34", "efficientnet_b0"])
    p.add_argument("--limit",    type=int, default=20,
                   help="Max folders to evaluate")
    p.add_argument("--scale",    type=float, default=0.25,
                   help="Downscale factor for saved debug images (saves disk)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.labels) as f:
        labels = json.load(f)

    print(f"Loading model from {args.model} …")
    model = build_model(args.backbone).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()
    print("Model loaded.\n")

    errors = []

    for i, (folder_name, params) in enumerate(labels.items()):
        if i >= args.limit:
            break

        folder = args.raw_root / folder_name
        if not folder.exists():
            continue
        images = list_images(folder)
        if not images:
            continue

        # Use the middle image
        img_path = images[len(images) // 2]
        raw = cv2.imread(str(img_path))
        if raw is None:
            continue

        h, w = raw.shape[:2]
        manual_cx = float(params["cx"])
        manual_cy = float(params["cy"])
        manual_r  = float(params["r"])

        pred_cx, pred_cy, pred_r = predict(model, raw)

        # Error in pixels
        err_cx = abs(pred_cx - manual_cx)
        err_cy = abs(pred_cy - manual_cy)
        err_r  = abs(pred_r  - manual_r)
        errors.append((err_cx, err_cy, err_r))

        print(
            f"{folder_name:20s} | "
            f"manual=({manual_cx:.0f},{manual_cy:.0f},r={manual_r:.0f}) | "
            f"pred=({pred_cx:.0f},{pred_cy:.0f},r={pred_r:.0f}) | "
            f"err=({err_cx:.1f},{err_cy:.1f},{err_r:.1f})"
        )

        # Draw
        vis_manual = draw_circle_on(raw, manual_cx, manual_cy, manual_r,
                                    colour=(0, 255, 0), label="MANUAL")
        vis_pred   = draw_circle_on(raw, pred_cx,   pred_cy,   pred_r,
                                    colour=(0, 0, 255), label="PRED")
        panel = side_by_side(vis_manual, vis_pred, target_h=600)

        # Optional: downscale for storage
        if args.scale < 1.0:
            ph, pw = panel.shape[:2]
            panel = cv2.resize(panel, (int(pw * args.scale), int(ph * args.scale)))

        out_path = args.out_dir / f"{i:03d}_{folder_name}.jpg"
        cv2.imwrite(str(out_path), panel)

    if errors:
        mean_cx = sum(e[0] for e in errors) / len(errors)
        mean_cy = sum(e[1] for e in errors) / len(errors)
        mean_r  = sum(e[2] for e in errors) / len(errors)
        print(f"\n--- Mean pixel errors over {len(errors)} folders ---")
        print(f"  cx: {mean_cx:.1f} px")
        print(f"  cy: {mean_cy:.1f} px")
        print(f"  r : {mean_r:.1f} px")
        print(f"\nDebug images saved to: {args.out_dir}")


if __name__ == "__main__":
    main()