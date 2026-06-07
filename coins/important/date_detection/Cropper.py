from PIL import Image
import os

# Folder path
input_folder = '/home/noah/projects/image-identifier/data/coins/PCGS/OriginalsUncropped/1983'
output_folder = '/home/noah/projects/image-identifier/data/coins/PCGS/cropped/1983'
left_half_folder = os.path.join(output_folder, "Heads")
right_half_folder = os.path.join(output_folder, "Tails")

# Create directories if they don't exist
os.makedirs(left_half_folder, exist_ok=True)
os.makedirs(right_half_folder, exist_ok=True)

# Process each JPEG image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):  # Focus only on JPEG files
        input_image_path = os.path.join(input_folder, filename)
        image = Image.open(input_image_path)

        # Check if the image dimensions are correct
        if image.size == (5757, 2905):
            # Crop the left and right halves
            left_half = image.crop((0, 0, 2878, 2905))
            right_half = image.crop((2878, 0, 5757, 2905))

            # Save the cropped images
            left_half.save(os.path.join(left_half_folder, f"Heads_{filename}"))
            right_half.save(os.path.join(right_half_folder, f"Tails_{filename}"))

            print(f"Processed and saved: {filename}")
        else:
            print(f"Skipped {filename}: Incorrect dimensions ({image.size})")
