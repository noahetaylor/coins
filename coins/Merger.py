import os
import zipfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import class_weight
from Categorize import findMiscategorizedImages
from ChartTraining import chartTraining

# Settings
image_size = 150
num_epochs = 30
batch_size = 32

# Paths 
base_dir = '/home/noah/projects/image-identifier/data/coins/Bicentennial'
train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'Testing')
miscategorized_folder = '/home/noah/projects/image-identifier/data/coins/Bicentennial/Miscategorized'

# Function to detect class imbalance
def get_class_distribution(directory):
    class_counts = {}
    for subdir in os.listdir(directory):
        subpath = os.path.join(directory, subdir)
        if os.path.isdir(subpath):
            class_counts[subdir] = len(os.listdir(subpath))
    return class_counts

# Detect class imbalance
train_class_counts = get_class_distribution(train_dir)
validation_class_counts = get_class_distribution(validation_dir)
print("Training class distribution:", train_class_counts)
print("Validation class distribution:", validation_class_counts)

# Generate class weights to handle imbalance
all_train_classes = []
for subdir, count in train_class_counts.items():
    all_train_classes.extend([subdir] * count)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_train_classes),
    y=all_train_classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Function to build and train the model
def trainModel():
    model = tf.keras.models.Sequential([
        # Initial convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Add a second convolutional block for more feature extraction
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Flatten and pass to dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Regularization to prevent overfitting
        tf.keras.layers.Dense(4, activation='softmax')  # Output layer for 4 classes
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Train the model
    trainingResults = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        class_weight=class_weights,
        verbose=2
    )

    return model, trainingResults

# Train model
print("=============================================")
print(" TRAIN MODEL")
print("=============================================")
model, trainingResults = trainModel()

# Find miscategorized images
print("=============================================")
print(" FIND MISCATEGORIZED IMAGES")
print("=============================================")
findMiscategorizedImages(model, validation_dir, miscategorized_folder, image_size)

# Chart training results
print("=============================================")
print(" CHART TRAINING RESULTS")
print("=============================================")
chartTraining(trainingResults)

# Plot metrics (as an additional visualization if needed)
# history = trainingResults.history
# epochs = range(1, num_epochs + 1)

# # Accuracy plot
# plt.plot(epochs, history['accuracy'], 'bo', label='Training Accuracy')
# plt.plot(epochs, history['val_accuracy'], 'b', label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.show()

# # Loss plot
# plt.plot(epochs, history['loss'], 'ro', label='Training Loss')
# plt.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()
