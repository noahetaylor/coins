import torch
import math
import cv2
from torchvision import models, transforms
from pathlib import Path
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMPORTANT_DIR = Path(__file__).resolve().parents[1]

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------
# LOAD FINE MODEL
# -----------------------------
def load_fine_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(
        IMPORTANT_DIR / "models" / "fine_rotation_model.pth",
        map_location=DEVICE
    ))
    model.to(DEVICE)
    model.eval()
    return model

FINE_MODEL = load_fine_model()

# -----------------------------
# PREDICT FINE ANGLE
# -----------------------------
def predict_fine(img):
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sin_pred, cos_pred = FINE_MODEL(img_t)[0].cpu().numpy()

    angle = math.degrees(math.atan2(sin_pred, cos_pred))
    if angle < 0:
        angle += 360
    return angle

# -----------------------------
# ROTATE WITH OPENCV
# -----------------------------
def rotate_cv(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))

# -----------------------------
# PROCESS A SINGLE FOLDER
# -----------------------------
def process_folder(folder_path, output_root):
    # Collect images
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    images = []
    for ext in exts:
        images.extend(folder_path.glob(ext))

    if len(images) == 0:
        return

    # Sort for consistency
    images = sorted(images)

    # Pick 5 evenly spaced images for averaging
    n = len(images)
    sample_indices = [
        n // 6,
        2 * n // 6,
        3 * n // 6,
        4 * n // 6,
        5 * n // 6
    ]

    sin_sum = 0.0
    cos_sum = 0.0

    print(f"[{folder_path.name}] samples:")

    for idx in sample_indices:
        img_path = images[idx]
        pil_img = Image.open(img_path).convert("RGB")
        angle = predict_fine(pil_img)

        # Log each sample
        print(f"  sample={img_path.name}, angle={angle:.2f}")

        # Convert to vector
        rad = math.radians(angle)
        sin_sum += math.sin(rad)
        cos_sum += math.cos(rad)

    # Average vector
    avg_rad = math.atan2(sin_sum, cos_sum)
    avg_angle = math.degrees(avg_rad)
    if avg_angle < 0:
        avg_angle += 360

    print(f"  AVERAGED angle={avg_angle:.2f}")

    # Compute rotation needed (inverse)
    rotation_needed = -avg_angle
    print(f"  ROTATE={rotation_needed:.2f}")

    # Output folder
    rel = folder_path.relative_to(INPUT_DIR)
    out_folder = output_root / rel
    out_folder.mkdir(parents=True, exist_ok=True)

    # Apply SAME rotation to all images in this folder
    for img_path in images:
        cv_img = cv2.imread(str(img_path))
        final_img = rotate_cv(cv_img, rotation_needed)

        out_path = out_folder / img_path.name
        cv2.imwrite(str(out_path), final_img)



# -----------------------------
# MAIN
# -----------------------------
INPUT_DIR = Path("/home/noah/projects/image-identifier/data/coins/2026Data")
OUTPUT_DIR = Path("/home/noah/projects/image-identifier/data/coins/RotatedDataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    for folder in INPUT_DIR.rglob("*"):
        if folder.is_dir():
            process_folder(folder, OUTPUT_DIR)

    print("Rotation pipeline complete.")

if __name__ == "__main__":
    main()
