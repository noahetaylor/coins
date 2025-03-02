import os
import zipfile
import tensorflow as tf
from Categorize import findMiscategorizedImages
from ChartTraining import chartTraining

image_size = 150
num_epochs = 30
base_dir = '/home/noah/projects/image-identifier/data/coins/Bicentennial'
train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'Testing')
miscatagorized_folder = '/home/noah/projects/image-identifier/data/coins/Bicentennial/Miscatagorized'

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
        tf.keras.layers.Dense(128, activation='relu'),  # Smaller dense layer for better generalization
        tf.keras.layers.Dropout(0.5),  # Regularization to prevent overfitting
        tf.keras.layers.Dense(4, activation='softmax')  # Output layer for 4 classes
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy'])


    # This code has changed. Now instead of the ImageGenerator just rescaling
    # the image, we also rotate and do other operations
    # Updated to do image augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        brightness_range=[0.8, 1.2],  # Adjust brightness
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )


    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            train_dir,  # This is the source directory for training images
            target_size=(image_size, image_size),  # All images will be resized to 150x150
            batch_size=100,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='categorical')

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(image_size, image_size),
            batch_size=12,
            class_mode='categorical')

    print(train_generator.class_indices)
    print(validation_generator.class_indices)
    class_weights = {0: 5.0, 1: 5.0, 2: 1.0, 3: 1.0}

    trainingResults = model.fit(
        train_generator,
        steps_per_epoch=6,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=6,
        verbose=2,
        class_weight=class_weights  # Add class_weight here
    )

    return model, trainingResults

print("=============================================")
print(" TRAIN MODEL")
print("=============================================")
model, trainingResults = trainModel()

print("=============================================")
print(" FIND MISCATAGORIZED IMAGES")
print("=============================================")
findMiscategorizedImages(model, validation_dir, miscatagorized_folder, image_size)

# print("=============================================")
# print(" CHART TRAINING RESULTS")
# print("=============================================")
# chartTraining(trainingResults)