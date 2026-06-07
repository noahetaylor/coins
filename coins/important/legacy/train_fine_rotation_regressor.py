import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import math

DATA_ROOT = Path("/home/noah/projects/image-identifier/data/coins/FineRotationDataset")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
IMPORTANT_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = IMPORTANT_DIR / "models" / "fine_rotation_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

class FineRotationDataset(Dataset):
    def __init__(self, root):
        self.files = list(root.glob("*.jpg"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")

        # Extract angle from filename
        angle = float(path.stem.split("_")[-1])
        rad = math.radians(angle)

        # Regression target: sinθ, cosθ
        target = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)

        return transform(img), target

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # sin, cos
    return model

def main():
    # Reproducibility
    torch.manual_seed(42)

    train_ds = FineRotationDataset(TRAIN_DIR)
    test_ds = FineRotationDataset(TEST_DIR)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = build_model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_test_loss = float('inf')

    EPOCHS = 16
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        train_loss = total_loss / len(train_ds)

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * imgs.size(0)

        test_loss = test_loss / len(test_ds) if len(test_ds) > 0 else 0.0

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f}  Test Loss: {test_loss:.6f}")

        # Save best model by test loss
        if test_loss and test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), OUT_PATH)
            print(f"Saved improved {OUT_PATH}")

    print("Training complete.")

if __name__ == "__main__":
    main()
