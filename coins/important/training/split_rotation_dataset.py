import os
import shutil
from pathlib import Path
import random

SOURCE = Path("/home/noah/projects/image-identifier/data/coins/2026RotationTraining")
DEST = Path("/home/noah/projects/image-identifier/data/coins/2026RotationTrainingSplit")

TRAIN_RATIO = 0.8  # 80% train, 20% test

def main():
    classes = ["rot_train_0", "rot_train_90", "rot_train_180", "rot_train_270"]

    for split in ["train", "test"]:
        for cls in classes:
            (DEST / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        files = list((SOURCE / cls).glob("*"))
        random.shuffle(files)

        split_idx = int(len(files) * TRAIN_RATIO)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for f in train_files:
            shutil.copy(f, DEST / "train" / cls / f.name)

        for f in test_files:
            shutil.copy(f, DEST / "test" / cls / f.name)

    print("Dataset split complete.")

if __name__ == "__main__":
    main()
