import os
import shutil


def prepare_verification_dataset(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    count = 0


    for year_folder in os.listdir(source_dir):
        year_path = os.path.join(source_dir, year_folder)
        if not os.path.isdir(year_path) or not year_folder.isdigit():
            continue  # Skip non-year folders


        for fname in os.listdir(year_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue


            src_path = os.path.join(year_path, fname)
            new_name = f"{year_folder}_{fname}"
            dst_path = os.path.join(target_dir, new_name)


            shutil.copy2(src_path, dst_path)
            count += 1


    print(f"✅ Copied and renamed {count} images into {target_dir}")

source_dir = "/home/noah/projects/image-identifier/data/coins/Bounds/Heads/Testing"
target_dir = "/home/noah/projects/image-identifier/data/coins/Bounds/Heads/date_verification"


prepare_verification_dataset(source_dir, target_dir)
