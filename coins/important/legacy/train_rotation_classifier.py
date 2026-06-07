import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ============================
# PATHS
# ============================
DATA_ROOT = Path("/home/noah/projects/image-identifier/data/coins/2026RotationTrainingSplit")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
IMPORTANT_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = IMPORTANT_DIR / "models" / "rotation_model.pth"

# ============================
# HYPERPARAMETERS
# ============================
BATCH_SIZE = 16
EPOCHS = 16
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# TRANSFORMS
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ============================
# DATA LOADERS
# ============================
def get_loaders():
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_ds.classes

# ============================
# MODEL
# ============================
def build_model(num_classes):
    # PRETRAINED RESNET18 — CRITICAL FIX
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ============================
# TRAINING LOOP
# ============================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# ============================
# TEST LOOP
# ============================
def test(model, loader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# ============================
# MAIN
# ============================
def main():
    train_loader, test_loader, classes = get_loaders()
    print("Classes:", classes)

    model = build_model(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = test(model, test_loader, criterion)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"  Test  Loss: {test_loss:.4f}  Test  Acc: {test_acc:.4f}")

    # Save model + class names
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
    }, OUT_PATH)

    print(f"Saved model to {OUT_PATH}")

if __name__ == "__main__":
    main()
