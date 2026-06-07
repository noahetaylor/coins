import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

from pathlib import Path

IMPORTANT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = IMPORTANT_DIR / "models" / "rotation_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model, classes

MODEL, CLASSES = load_model()

def predict_angle(pil_img):
    img_t = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(img_t)
        _, pred = outputs.max(1)
    label = CLASSES[pred.item()]

    # Convert folder name → numeric angle
    if "0" in label:
        return 0
    if "90" in label:
        return 90
    if "180" in label:
        return 180
    if "270" in label:
        return 270

    raise ValueError(f"Unrecognized label: {label}")

def rotate_image(input_path, output_path):
    pil_img = Image.open(input_path).convert("RGB")
    angle = predict_angle(pil_img)

    rotated = pil_img.rotate(-angle, expand=True)
    rotated.save(output_path)
    print(f"{input_path} -> {output_path} (angle {angle})")

if __name__ == "__main__":
    import sys
    inp = sys.argv[1]
    out = sys.argv[2]
    rotate_image(inp, out)
