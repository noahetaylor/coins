import torch
import math
from pathlib import Path
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMPORTANT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = IMPORTANT_DIR / "models" / "fine_rotation_model.pth"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

MODEL = load_model()

def predict_angle(img):
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sin_pred, cos_pred = MODEL(img_t)[0].cpu().numpy()

    angle = math.degrees(math.atan2(sin_pred, cos_pred))
    if angle < 0:
        angle += 360
    return angle

# --- FIXED TEST PATH ---
test_path = "/home/noah/projects/image-identifier/data/coins/FineRotationDataset/1979_D_img_0056_96.70.jpg"

img = Image.open(test_path).convert("RGB")
pred = predict_angle(img)

print(f"Predicted angle: {pred:.2f}°")
