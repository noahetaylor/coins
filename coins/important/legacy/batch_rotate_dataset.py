import os
from pathlib import Path
from rotate_with_model import rotate_image

INPUT_ROOT = Path("/home/noah/projects/image-identifier/data/coins/2026Data")
OUTPUT_ROOT = Path("/home/noah/projects/image-identifier/data/coins/2026Data_rotated")

def main():
    for root, dirs, files in os.walk(INPUT_ROOT):
        rel = Path(root).relative_to(INPUT_ROOT)
        out_dir = OUTPUT_ROOT / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            in_path = Path(root) / f
            out_path = out_dir / f

            if out_path.exists():
                continue

            rotate_image(str(in_path), str(out_path))

if __name__ == "__main__":
    main()
