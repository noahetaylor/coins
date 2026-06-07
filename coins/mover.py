import os
import shutil


# 🔍 Paths
base_dir = "/home/noah/projects/image-identifier/data/coins/Bounds/Heads"
labels_dir = os.path.join(base_dir, "Testing", "HeadsQuartersAnnotations")
images_root = os.path.join(base_dir, "Testing")


# 🧠 Step 1: Get all label files
label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]


moved = 0
missing_images = []


# 🚀 Step 2: For each label, find the correct image and move the label
for label_file in label_files:
    label_name = os.path.splitext(label_file)[0]  # Remove .txt extension
    possible_image = label_name + ".jpg"


    # Search for the matching image in year folders
    for year_folder in os.listdir(images_root):
        year_path = os.path.join(images_root, year_folder)
        image_path = os.path.join(year_path, possible_image)


        if os.path.isdir(year_path) and os.path.exists(image_path):
            destination = os.path.join(year_path, label_file)
            source = os.path.join(labels_dir, label_file)


            shutil.move(source, destination)
            moved += 1
            break
    else:
        missing_images.append(label_file)


# ✅ Summary
print(f"\n✅ Moved {moved} label files successfully.")
if missing_images:
    print(f"\n⚠️ WARNING: Could not find matching images for {len(missing_images)} labels:")
    for label in missing_images:
        print(f" - {label}")
else:
    print("🎉 All labels matched and were moved!")
