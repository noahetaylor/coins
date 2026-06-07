import os
import shutil
import random

# Folder paths
source_folder = '/home/noah/projects/image-identifier/data/coins/PCGS/Heads/Training/BicentennialsP'
destination_folder = '/home/noah/projects/image-identifier/data/coins/PCGS/Heads/Testing/BicentennialsP'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Get a list of all JPEG files in the source folder
images = [f for f in os.listdir(source_folder) if f.lower().endswith(".jpg")]

# Calculate 1/10th of the images
num_images_to_move = max(5, len(images) // 10)  # Ensure at least five images are moved
images_to_move = random.sample(images, num_images_to_move)

# Move the selected images
for image in images_to_move:
    source_path = os.path.join(source_folder, image)
    destination_path = os.path.join(destination_folder, image)
    shutil.move(source_path, destination_path)
    print(f"Moved: {image}")

print(f"{len(images_to_move)} JPEG images moved to {destination_folder}.")
