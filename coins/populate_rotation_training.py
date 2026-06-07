import os
import random
import shutil
from pathlib import Path

# Paths
DATA_ROOT = Path("/home/noah/projects/image-identifier/data/coins/2026Data")
ROT_ROOT = Path("/home/noah/projects/image-identifier/data/coins/2026RotationTraining")

# Rotation class folders
ROT_CLASSES = [
    "rot_train_0",
    "rot_train_90",
    "rot_train_180",
    "rot_train_270"
]

# How many images to copy from each coin folder
IMAGES_PER_FOLDER = 3

# Maximum images allowed in each rotation folder
MAX_IMAGES = 20

def copy_samples():
    # Ensure rotation folders exist
    for cls in ROT_CLASSES:
        (ROT_ROOT / cls).mkdir(parents=True, exist_ok=True)

    # Loop through each coin folder
    for folder in DATA_ROOT.iterdir():
        if not folder.is_dir():
            continue

        # Get all image files
        images = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if len(images) == 0:
            continue

        # Randomly pick N images
        sample = random.sample(images, min(IMAGES_PER_FOLDER, len(images)))

        print(f"Folder {folder.name}: copying {len(sample)} images")

        # Copy each selected image into each rotation class folder
        for img in sample:
            for cls in ROT_CLASSES:
                dest = ROT_ROOT / cls / f"{folder.name}_{img.name}"
                shutil.copy(img, dest)

def trim_rotation_folders():
    for cls in ROT_CLASSES:
        folder = ROT_ROOT / cls
        files = [f for f in folder.iterdir() if f.is_file()]

        if len(files) <= MAX_IMAGES:
            print(f"{cls}: {len(files)} images (no trimming needed)")
            continue

        # Randomly choose which ones to keep
        keep = set(random.sample(files, MAX_IMAGES))

        # Delete the rest
        for f in files:
            if f not in keep:
                print(f"Deleting extra file: {f}")
                f.unlink()

        print(f"{cls}: trimmed to {MAX_IMAGES} images")

def main():
    copy_samples()
    trim_rotation_folders()

if __name__ == "__main__":
    main()
