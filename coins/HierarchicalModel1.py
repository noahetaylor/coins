import os
import zipfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Categories
broadCategory = 'HeadsVsTails'
fineCategory = 'QuartersVsBicentennials'

# Unzip dataset
local_zip = '/tmp/coins/qtest9.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/coins')
zip_ref.close()

# Paths
base_dir = '/tmp/coins/qtest9'
train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'Testing')

# Image settings
image_size = 150
batch_size = 32
num_epochs = 15

# Model definition
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

# Define model
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

# Data generators
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

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Metrics and plotting
print("Available history keys:", history.history.keys())

# Define epochs
epochs = range(1, num_epochs + 1)

# Extract metrics
heads_vs_tails_acc = history.history['HeadsVsTails_accuracy']
heads_vs_tails_val_acc = history.history['val_HeadsVsTails_accuracy']
quarters_vs_bicentennials_acc = history.history['QuartersVsBicentennials_accuracy']
quarters_vs_bicentennials_val_acc = history.history['val_QuartersVsBicentennials_accuracy']

heads_vs_tails_loss = history.history['HeadsVsTails_loss']
heads_vs_tails_val_loss = history.history['val_HeadsVsTails_loss']
quarters_vs_bicentennials_loss = history.history['QuartersVsBicentennials_loss']
quarters_vs_bicentennials_val_loss = history.history['val_QuartersVsBicentennials_loss']

# Accuracy plot
plt.plot(epochs, heads_vs_tails_acc, 'bo', label='Heads vs Tails Training Accuracy')
plt.plot(epochs, heads_vs_tails_val_acc, 'b', label='Heads vs Tails Validation Accuracy')
plt.plot(epochs, quarters_vs_bicentennials_acc, 'ro', label='Quarters vs Bicentennials Training Accuracy')
plt.plot(epochs, quarters_vs_bicentennials_val_acc, 'r', label='Quarters vs Bicentennials Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Loss plot
plt.plot(epochs, heads_vs_tails_loss, 'bo', label='Heads vs Tails Training Loss')
plt.plot(epochs, heads_vs_tails_val_loss, 'b', label='Heads vs Tails Validation Loss')
plt.plot(epochs, quarters_vs_bicentennials_loss, 'ro', label='Quarters vs Bicentennials Training Loss')
plt.plot(epochs, quarters_vs_bicentennials_val_loss, 'r', label='Quarters vs Bicentennials Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

