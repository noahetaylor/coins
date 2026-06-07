import os
import shutil


# ----------- ✅ CONFIGURE PATHS -----------
source_dir = "/home/noah/projects/image-identifier/data/coins/digitsannotations/date_cropmatchedbase222"  # Where all digit crops live
output_dir = "/home/noah/projects/image-identifier/data/coins/digitsannotations/date_cropmatchedout222"  # Where grouped pairs go


os.makedirs(output_dir, exist_ok=True)


# ----------- ✅ GROUP MATCHED FILES -----------
count = 0
for filename in os.listdir(source_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        base = os.path.splitext(filename)[0]
        label_file = base + ".txt"
        label_path = os.path.join(source_dir, label_file)


        if os.path.exists(label_path):
            # Copy image + label to output_dir
            img_src = os.path.join(source_dir, filename)
            label_src = label_path


            img_dst = os.path.join(output_dir, filename)
            label_dst = os.path.join(output_dir, label_file)


            shutil.copy(img_src, img_dst)
            shutil.copy(label_src, label_dst)
            count += 1


print(f"✅ Grouped and copied {count} image/label pairs to:", output_dir)
