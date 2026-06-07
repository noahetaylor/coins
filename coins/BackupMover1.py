import os
import shutil


# 📁 Configuration
base_dir = "/home/noah/projects/image-identifier/data/coins/Bounds/Heads"
folders_to_update = ["Training", "Testing"]  # You can include other folders too
backup_dir = os.path.join(base_dir, "OriginalLabelsBackup")


# 🚀 Create backup directory
os.makedirs(backup_dir, exist_ok=True)


def flatten_label_ids(folder):
    full_path = os.path.join(base_dir, folder)
    for year in os.listdir(full_path):
        year_path = os.path.join(full_path, year)
        if not os.path.isdir(year_path):
            continue


        for fname in os.listdir(year_path):
            if fname.endswith(".txt"):
                label_path = os.path.join(year_path, fname)


                # 📦 Backup original label
                rel_backup_path = os.path.join(backup_dir, folder, year)
                os.makedirs(rel_backup_path, exist_ok=True)
                shutil.copy(label_path, os.path.join(rel_backup_path, fname))


                # 🧹 Read and replace class_id with 0
                with open(label_path, "r") as f:
                    lines = f.readlines()


                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        parts[0] = "0"  # Set class_id to 0
                        new_lines.append(" ".join(parts))


                # ✍️ Overwrite with updated label
                with open(label_path, "w") as f:
                    f.write("\n".join(new_lines) + "\n")


print("\n🔁 Updating label files...")
for folder in folders_to_update:
    flatten_label_ids(folder)


print(f"\n✅ All label files updated to use class ID = 0.")
print(f"🗂️ Backup of original labels saved in: {backup_dir}")
