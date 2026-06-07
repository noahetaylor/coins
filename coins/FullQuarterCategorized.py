import os
import zipfile
import tensorflow as tf
from Categorize import findMiscategorizedImages
from ChartTraining import chartTraining
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt

image_size = 150
num_epochs = 30
base_dir = '/home/noah/projects/image-identifier/data/coins/AIHierarchy'
train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'Testing')
miscatagorized_folder = '/home/noah/projects/image-identifier/data/coins/AIHierarchy/Miscatagorized'
batch_size = 32

broadCategory = 'HeadsVsTails'
fineCategory = 'QuartersVsBicentennials'

# Function to detect class imbalance
def get_class_distribution(directory):
    class_counts = {}
    for subdir in os.listdir(directory):
        subpath = os.path.join(directory, subdir)
        if os.path.isdir(subpath):
            class_counts[subdir] = len(os.listdir(subpath))
    return class_counts


def chartHierarchyTrainingData(trainingResults):
    # Plot metrics (optional additional visualization)
    history = trainingResults.history
    epochs = range(1, num_epochs + 1)

    # Accuracy plot
    plt.plot(epochs, history['HeadsVsTails_accuracy'], 'bo', label='Heads vs Tails Training Accuracy')
    plt.plot(epochs, history['val_HeadsVsTails_accuracy'], 'b', label='Heads vs Tails Validation Accuracy')
    plt.plot(epochs, history['QuartersVsBicentennials_accuracy'], 'ro', label='Quarters vs Bicentennials Training Accuracy')
    plt.plot(epochs, history['val_QuartersVsBicentennials_accuracy'], 'r', label='Quarters vs Bicentennials Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    # Loss plot
    plt.plot(epochs, history['HeadsVsTails_loss'], 'bo', label='Heads vs Tails Training Loss')
    plt.plot(epochs, history['val_HeadsVsTails_loss'], 'b', label='Heads vs Tails Validation Loss')
    plt.plot(epochs, history['QuartersVsBicentennials_loss'], 'ro', label='Quarters vs Bicentennials Training Loss')
    plt.plot(epochs, history['val_QuartersVsBicentennials_loss'], 'r', label='Quarters vs Bicentennials Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def trainModel():
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

    # Hierarchical model definition
    input_layer = tf.keras.Input(shape=(image_size, image_size, 3))

    # Shared base layers
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)

    # Broad category branch
    branch_1 = tf.keras.layers.Dense(128, activation='relu')(x)
    branch_1 = tf.keras.layers.Dense(2, activation='softmax', name=broadCategory)(branch_1)

    # Fine-grained category branch
    branch_2 = tf.keras.layers.Dense(128, activation='relu')(x)
    branch_2 = tf.keras.layers.Dense(2, activation='softmax', name=fineCategory)(branch_2)

    # Define hierarchical model
    model = tf.keras.Model(inputs=input_layer, outputs=[branch_1, branch_2])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            broadCategory: 'categorical_crossentropy',
            fineCategory: 'categorical_crossentropy'
        },
        metrics={
            broadCategory: 'accuracy',
            fineCategory: 'accuracy'
        }
    )

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

print("=============================================")
print(" TRAIN MODEL")
print("=============================================")
model, trainingResults = trainModel()

print("=============================================")
print(" CHART TRAINING RESULTS")
print("=============================================")
# chartTraining(trainingResults)
chartHierarchyTrainingData(trainingResults)


# print("=============================================")
# print(" FIND MISCATAGORIZED IMAGES")
# print("=============================================")
# findMiscategorizedImages(model, validation_dir, miscatagorized_folder, image_size)

