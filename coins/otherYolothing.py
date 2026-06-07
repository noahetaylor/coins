import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from ultralytics import YOLO  # For object detection

# Configure TensorFlow threading
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.get_logger().setLevel('ERROR')
np.set_printoptions(suppress=True)

# GPU memory management
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Paths and Configuration
image_size = 300
num_epochs = 50
base_dir = '/home/noah/projects/image-identifier/data/coins/FourCategories'
train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'Testing')
miscategorizeFolder = os.path.join(base_dir, 'Miscategorized')

# Load YOLOv8 model for date detection
date_detection_model = YOLO('yolov8s-date.pt')

# Function to detect and crop the date region
def crop_date(image_path):
    img = cv2.imread(image_path)
    results = date_detection_model.predict(image_path)
    if results[0].boxes:
        x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])  # Get bounding box
        cropped_date = img[y1:y2, x1:x2]  # Crop detected date region
        return cropped_date
    return img  # Return original if no detection

# Function to process images
def preprocess_image(image_path):
    cropped_image = crop_date(image_path)
    resized_image = cv2.resize(cropped_image, (image_size, image_size))
    normalized_image = resized_image / 255.0  # Normalize pixel values
    return normalized_image

# Data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, fill_mode='reflect'
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Training and validation generators with preprocessing
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(image_size, image_size), batch_size=50, class_mode='categorical'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(image_size, image_size), batch_size=20, class_mode='categorical'
)

# Model architecture (CNN)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')  # Normal vs Bicentennial classification
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=validation_generator, epochs=num_epochs, verbose=2)

# Function to visualize results
def plot_training_results(history):
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    loss, val_loss = history.history['loss'], history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_training_results(history)

# Function to find miscategorized images
def findMiscategorizedImages(model, test_dir, output_dir):
    shutil.rmtree(output_dir, True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    num_images, num_errors = 0, 0
    categories = sorted([f for f in Path(test_dir).iterdir() if f.is_dir()])
    
    for category in categories:
        for file in category.iterdir():
            num_images += 1
            img_array = preprocess_image(str(file))
            img_array = np.expand_dims(img_array, axis=0)
            
            predicted_class = np.argmax(model.predict(img_array)[0])
            predicted_category = categories[predicted_class].name
            
            if category.name != predicted_category:
                num_errors += 1
                print(f"[{category.name}] '{file.name}' misclassified as [{predicted_category}]")
                shutil.copy(str(file), os.path.join(output_dir, f"{predicted_category}-{file.name}"))
    
    print("Final Accuracy:", (num_images - num_errors) / num_images)

findMiscategorizedImages(model, validation_dir, miscategorizeFolder)
