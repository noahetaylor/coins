import os
import tensorflow as tf
from Categorize import findMiscategorizedImages
from sklearn.utils import class_weight
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# **Fix 1: Configure GPU Memory Management**
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print(f"Could not set memory growth: {e}")

# **Fix 2: Limit TensorFlow Threading**
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# **Fix 3: Suppress Verbose Logs**
tf.get_logger().setLevel('ERROR')
np.set_printoptions(suppress=True)

# Paths and Configuration
image_size = 150
num_epochs = 20
base_dir = '/home/noah/projects/image-identifier/data/coins/HeadsTails'
train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'Testing')
miscategorizeFolder = os.path.join(base_dir, 'Miscategorized')

# **Detect Class Distribution**
def get_class_distribution(directory):
    class_counts = {}
    for subdir in os.listdir(directory):
        subpath = os.path.join(directory, subdir)
        if os.path.isdir(subpath):
            class_counts[subdir] = len(os.listdir(subpath))
    return class_counts

# Print class distributions
train_class_counts = get_class_distribution(train_dir)
validation_class_counts = get_class_distribution(validation_dir)
print("Training class distribution:", train_class_counts)
print("Validation class distribution:", validation_class_counts)

# **Generate Class Weights**
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

# **Image Preprocessing Function**
def preprocess_image(image):
    try:
        if isinstance(image, str):
            img = tf.keras.utils.load_img(image, target_size=None)
            img_array = tf.keras.utils.img_to_array(img)
        elif isinstance(image, np.ndarray):
            img_array = image
        else:
            raise ValueError("Invalid input type for preprocess_image.")

        img_array = tf.image.resize_with_pad(img_array, target_height=image_size, target_width=image_size)
        return img_array / 255.0
    except Exception as e:
        print(f"Error processing image: {image}, error: {e}")
        return np.zeros((image_size, image_size, 3)) / 255.0

# **Training and Validation Data Augmentation**
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_image
)

# **Verify Dataset File Paths**
def check_file_paths(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")

check_file_paths(train_dir)
check_file_paths(validation_dir)

# **Visualize Augmented Images**
def visualize_augmentation(generator, directory):
    augmented_generator = generator.flow_from_directory(
        directory,
        target_size=(image_size, image_size),
        batch_size=5,
        class_mode='binary'
    )
    augmented_images, _ = next(augmented_generator)
    for i, img in enumerate(augmented_images):
        plt.imshow(img)
        plt.title(f"Augmented Image {i+1}")
        plt.axis('off')
        plt.show()  # Display the plot interactively

print("Visualizing Augmentation...")
visualize_augmentation(train_datagen, train_dir)

# **Prepare Training and Validation Generators**
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=16,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=8,
    class_mode='binary'
)

print(train_generator.class_indices)
print(validation_generator.class_indices)

# **Model Architecture**
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# **Train the Model**
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 16,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 8,
    verbose=2,
    class_weight=class_weights
)

# **Display Training Results**
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

    plt.show()  # Display the plot instead of saving it

print("Plotting Training Results...")
plot_training_results(history)

# **Analyze Miscategorized Images**
findMiscategorizedImages(model, validation_dir, miscategorizeFolder, image_size)
