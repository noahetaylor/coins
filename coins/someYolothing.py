import cv2
import os
import numpy as np
import tensorflow as tf
from Categorize import findMiscategorizedImages
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# Additional import for object detection
from yolov5 import detect  # Ensure YOLOv5 is installed and configured

# Configure TensorFlow threading
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

tf.get_logger().setLevel('ERROR')
np.set_printoptions(suppress=True)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Paths and Configuration
image_size = 500
num_epochs = 50
base_dir = '/home/noah/projects/image-identifier/data/coins/PCGS/Heads'
train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'Testing')
miscategorizeFolder = os.path.join(base_dir, 'Miscategorized')

# New directories for cropped images
cropped_train_dir = os.path.join(base_dir, 'Cropped_Training')
cropped_validation_dir = os.path.join(base_dir, 'Cropped_Testing')
os.makedirs(cropped_train_dir, exist_ok=True)
os.makedirs(cropped_validation_dir, exist_ok=True)

# Function to crop dates using YOLO
def crop_date(image_path, output_dir):
    try:
        # Run YOLO model to detect date region
        results = detect(image_path)  # Ensure this runs YOLO detection
        if results is None or len(results) == 0:
            print(f"No date detected for {image_path}")
            return

        # Iterate through detected bounding boxes
        image = cv2.imread(image_path)
        for box in results:
            x, y, w, h = box  # Extract bounding box coordinates
            cropped = image[y:y + h, x:x + w]
            
            # Save the cropped date region
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, cropped)
            print(f"Cropped and saved: {output_path}")
    except Exception as e:
        print(f"Error cropping date: {e}")

# Apply cropping to datasets
def preprocess_and_crop_dataset(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            image_path = os.path.join(root, file)
            crop_date(image_path, output_dir)

preprocess_and_crop_dataset(train_dir, cropped_train_dir)
preprocess_and_crop_dataset(validation_dir, cropped_validation_dir)

# Data augmentation for cropped images
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Prepare training and validation generators for cropped images
train_generator = train_datagen.flow_from_directory(
    cropped_train_dir,
    target_size=(image_size, image_size),
    batch_size=50,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    cropped_validation_dir,
    target_size=(image_size, image_size),
    batch_size=20,
    class_mode='categorical'
)

print(train_generator.class_indices)
print(validation_generator.class_indices)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(18, activation='softmax')  # 18 classes for coin dates
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with cropped images
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 50,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 20,
    verbose=2
)

# Visualize training results
def plot_training_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
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

print("Plotting Training Results...")
plot_training_results(history)

# Analyze miscategorized images
findMiscategorizedImages(model, cropped_validation_dir, miscategorizeFolder, image_size)
