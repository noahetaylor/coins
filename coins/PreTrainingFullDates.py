import os
import tensorflow as tf
from Categorize import findMiscategorizedImages
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt

# Configure TensorFlow threading
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Suppress TensorFlow verbosity
tf.get_logger().setLevel('ERROR')

# Suppress NumPy array printouts
np.set_printoptions(suppress=True)

# Configure GPU memory growth to prevent "unaligned tcache chunk" errors
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

# Function to detect class imbalance
# def get_class_distribution(directory):
#     class_counts = {}
#     for subdir in os.listdir(directory):
#         subpath = os.path.join(directory, subdir)
#         if os.path.isdir(subpath):
#             class_counts[subdir] = len(os.listdir(subpath))
#     return class_counts

# # Detect class imbalance
# train_class_counts = get_class_distribution(train_dir)
# validation_class_counts = get_class_distribution(validation_dir)
# print("Training class distribution:", train_class_counts)
# print("Validation class distribution:", validation_class_counts)


# # Generate class weights to handle imbalance
# all_train_classes = []
# for subdir, count in train_class_counts.items():
#     all_train_classes.extend([subdir] * count)
# class_weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(all_train_classes),
#     y=all_train_classes
# )
# class_weights = dict(enumerate(class_weights))
# print("Class weights:", class_weights)




# Function to preprocess images
# def preprocess_image(image):
#     try:
#         if isinstance(image, str):  # If input is a file path
#             img = tf.keras.utils.load_img(image, target_size=None)
#             img_array = tf.keras.utils.img_to_array(img)
#         elif isinstance(image, np.ndarray):  # If input is already an array
#             img_array = image
#         else:
#             raise ValueError("Invalid input type for preprocess_image.")

#         # Resize with padding to preserve aspect ratio
#         img_array = tf.image.resize_with_pad(img_array, target_height=image_size, target_width=image_size)

#         # Normalize for training
#         return img_array / 255.0
#     except Exception as e:
#         print(f"Error processing image: {e}")
#         return np.zeros((image_size, image_size, 3)) / 255.0  # Return blank image on failure


# Data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #preprocessing_function=preprocess_image,
    rescale=1./255,
    #rotation_range=10,  # Reduced rotation
    #width_shift_range=0.05,  # Subtle horizontal shift
    #height_shift_range=0.05,  # Subtle vertical shift
    #zoom_range=0.05,  # Subtle zoom effect
    #brightness_range=[0.95, 1.05],  # Narrowed brightness adjustment
    fill_mode='reflect'  # Smoother edge handling
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    # preprocessing_function=preprocess_image
)


# Verify dataset file paths
def check_file_paths(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")

check_file_paths(train_dir)
check_file_paths(validation_dir)

# Visualization function
def visualize_augmentation(generator, directory):
    generator_flow = generator.flow_from_directory(
        directory,
        target_size=(image_size, image_size),
        batch_size=8,  # Ensures at least 8 images are generated in one batch
        class_mode='categorical',  # Use 'categorical' if you have multiple classes
        keep_aspect_ratio=True
    )
    # Get a single batch of augmented images
    augmented_images, _ = next(generator_flow)

    # Display exactly 8 images
    for i in range(2):  # Limit to 8 images
        plt.imshow((augmented_images[i] * 255).astype('uint8'))  # Convert back to [0, 255] for display
        plt.title(f"Augmented Image {i + 1}")
        plt.axis('off')
        plt.show()


# Prepare training and validation generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=16,  # Reduced batch size for memory efficiency
    class_mode='categorical',
    keep_aspect_ratio=True
)
visualize_augmentation(train_datagen, train_dir)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=8,  # Further reduced batch size
    class_mode='categorical',
    keep_aspect_ratio=True
)

print(train_generator.class_indices)
print(validation_generator.class_indices)


# Load the base ResNet50 model with pretrained weights
base_model = tf.keras.applications.ResNet50(
    input_shape=(image_size, image_size, 3),  # Match the input size of your images
    include_top=False,  # Exclude the top fully connected layers
    weights='imagenet'  # Load ImageNet-pretrained weights
)

# Freeze the base model's layers to prevent them from being updated during initial training
base_model.trainable = True

# Create the full model
model = tf.keras.Sequential([
    base_model,  # Pretrained ResNet50 as the feature extractor
    tf.keras.layers.GlobalAveragePooling2D(),  # Reduce spatial dimensions
    tf.keras.layers.Dense(512, activation='relu'),  # Fully connected layer
    tf.keras.layers.Dropout(0.5),  # Dropout to prevent overfitting
    tf.keras.layers.Dense(18, activation='softmax')  # Final layer for 18 coin date classes
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),  # Low learning rate for fine-tuning
    loss='categorical_crossentropy',  # Multi-class classification
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    verbose=2  # Display training progress
)

# Visualize the training results
def plot_training_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(10, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

# Plot the training history
print("Plotting Training Results...")
plot_training_results(history)

# Analyze miscategorized images
findMiscategorizedImages(model, validation_dir, miscategorizeFolder, image_size)