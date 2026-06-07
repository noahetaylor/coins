import os
import cv2
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import numpy as np


# 📁 Set your test folder path here
test_folder = "/home/noah/projects/image-identifier/data/coins/Bounds/Heads/Testing"
output_dir = "diagnostic_output"
os.makedirs(output_dir, exist_ok=True)


# 🧠 List all test images to run OCR on
image_paths = []
for root, _, files in os.walk(test_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".png")):
            image_paths.append(os.path.join(root, file))


# 🔍 Run OCR on each image
for img_path in image_paths:
    print(f"\n🧪 Testing: {img_path}")
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)


    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    ocr_data = pytesseract.image_to_data(resized, config=config, output_type=Output.DICT)


    overlay = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    found = False


    for i in range(len(ocr_data["text"])):
        text = ocr_data["text"][i]
        conf = ocr_data["conf"][i]


        if text.strip().isdigit() and conf > 0:
            found = True
            x, y = ocr_data["left"][i], ocr_data["top"][i]
            w, h = ocr_data["width"][i], ocr_data["height"][i]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(overlay, f"{text} ({conf})", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


    # 🖼️ Save overlay + print raw OCR
    file_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir, f"{file_name}_overlay.jpg"), overlay)
    extracted = pytesseract.image_to_string(resized, config=config)
    print(f"📜 Raw OCR Output: {extracted.strip() if extracted else 'NONE'}")
    if not found:
        print("⚠️ No digit boxes detected in overlay.")


print("\n✅ Diagnostic run complete. Check your 'diagnostic_output/' folder for overlays.")
