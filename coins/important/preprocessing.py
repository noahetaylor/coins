import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


IMPORTANT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = IMPORTANT_DIR.parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "coins"
RAW_ROOT = DATA_ROOT / "2026Data"
ROTATED_ROOT = DATA_ROOT / "RotatedDataset"
LABELS_PATH = IMPORTANT_DIR / "labels" / "crop_labels.json"
MODELS_DIR = IMPORTANT_DIR / "models"
CROP_MODEL_PATH = MODELS_DIR / "crop_model.pth"
FINE_ROTATION_MODEL_PATH = MODELS_DIR / "rotation_v3.pth"   # ← updated from fine_rotation_full.pth
DEVICE = "cpu"                                               # ← forced CPU; don't risk the fanless 3060
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# Used for crop model and any legacy code — NOT used for rotation prediction
IMAGE_NET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Used for the v3 rotation model — 112px input
ROTATION_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),                           # ← was 224 in v1/v2
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float

    @classmethod
    def from_mapping(cls, params):
        return cls(float(params["cx"]), float(params["cy"]), float(params["r"]))


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def sample_evenly(items: Sequence[Path], count: int) -> List[Path]:
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def clamp_circle(circle: Circle, width: int, height: int, min_radius_ratio: float = 0.05) -> Circle:
    min_r = min(width, height) * min_radius_ratio
    max_r = min(width, height) * 0.5
    cx = float(np.clip(circle.cx, 0, width - 1))
    cy = float(np.clip(circle.cy, 0, height - 1))
    r = float(np.clip(circle.r, min_r, max_r))
    return Circle(cx, cy, r)


def crop_to_coin(
    img_cv: np.ndarray,
    circle: Circle,
    output_size: int = 512,
    padding: float = 0.0,
    background=(255, 255, 255),
    circular_mask: bool = False,
) -> np.ndarray:
    h, w = img_cv.shape[:2]
    circle = clamp_circle(circle, w, h)
    radius = circle.r * (1.0 + padding)

    x1 = int(math.floor(circle.cx - radius))
    y1 = int(math.floor(circle.cy - radius))
    x2 = int(math.ceil(circle.cx + radius))
    y2 = int(math.ceil(circle.cy + radius))

    out_side = max(x2 - x1, y2 - y1, 1)
    canvas = np.full((out_side, out_side, 3), background, dtype=np.uint8)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)
    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return cv2.resize(img_cv, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    canvas[dst_y1:dst_y1 + (src_y2 - src_y1), dst_x1:dst_x1 + (src_x2 - src_x1)] = img_cv[src_y1:src_y2, src_x1:src_x2]

    if circular_mask:
        mask = np.zeros((out_side, out_side), dtype=np.uint8)
        center = (int(round(circle.cx - x1)), int(round(circle.cy - y1)))
        cv2.circle(mask, center, int(round(radius)), 255, -1)
        canvas[mask == 0] = background

    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_CUBIC)


def rotate_image(img_cv: np.ndarray, angle_deg: float, background=(255, 255, 255)) -> np.ndarray:
    h, w = img_cv.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(
        img_cv,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )


def circular_mean_deg(angles: Iterable[float]) -> float:
    angles = list(angles)
    if not angles:
        raise ValueError("Cannot average an empty angle list")
    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles)
    angle = math.degrees(math.atan2(sin_sum, cos_sum))
    return angle + 360.0 if angle < 0 else angle


def draw_circle_overlay(img_cv: np.ndarray, circle: Circle, color=(0, 0, 255)) -> np.ndarray:
    vis = img_cv.copy()
    cv2.circle(vis, (int(round(circle.cx)), int(round(circle.cy))), int(round(circle.r)), color, 3)
    cv2.circle(vis, (int(round(circle.cx)), int(round(circle.cy))), 7, (0, 255, 0), -1)
    return vis


# ---------------------------------------------------------------------------
# Rotation model — v3 (MobileNetV3-Small, 112px input)
# ---------------------------------------------------------------------------

class RotationModel(nn.Module):
    """
    MobileNetV3-Small with a sin/cos regression head.
    Trained by train_rotation_model_v3_cpu.py at 112×112 input.
    """
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)
        in_features = base.classifier[0].in_features
        base.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        self.net = base

    def forward(self, x):
        return self.net(x)


def load_rotation_model(path: Path = FINE_ROTATION_MODEL_PATH) -> RotationModel:
    model = RotationModel()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_rotation_angle(model: RotationModel, img_cv: np.ndarray) -> float:
    """
    Predict the rotation angle of a cropped coin image in degrees [0, 360).
    Apply -angle as the correction to rotate the coin upright.
    """
    img_t = ROTATION_TRANSFORM(cv_to_pil(img_cv)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sin_pred, cos_pred = model(img_t)[0].cpu().numpy()
    angle = math.degrees(math.atan2(float(sin_pred), float(cos_pred)))
    return angle + 360.0 if angle < 0 else angle


# ---------------------------------------------------------------------------
# Crop model — v2 (ResNet-18, 224px input) — unchanged
# ---------------------------------------------------------------------------

class CropModel(nn.Module):
    """ResNet-18 backbone predicting normalised (cx, cy, r)."""
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 3)

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))


def load_crop_model(path: Path = CROP_MODEL_PATH) -> CropModel:
    model = CropModel()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_crop_circle(model: CropModel, img_cv: np.ndarray) -> Circle:
    h, w = img_cv.shape[:2]
    img_t = IMAGE_NET_TRANSFORM(cv_to_pil(img_cv)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cx_norm, cy_norm, r_norm = model(img_t)[0].cpu().numpy()
    circle = Circle(float(cx_norm) * w, float(cy_norm) * h, float(r_norm) * max(w, h))
    return clamp_circle(circle, w, h)