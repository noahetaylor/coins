import tensorflow as tf
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
#from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import confusion_matrix, classification_report
import shutil
import matplotlib.pyplot as plt

# Paths
train_dir = '/home/noah/projects/image-identifier/data/coins/OriginalYay/Training'
validation_dir = '/home/noah/projects/image-identifier/data/coins/OriginalYay/Testing'
image_size = 150
batch_size = 32
num_epochs = 10

# Broad category: Heads vs Tails
broad_train_dir = os.path.join(train_dir, 'HeadsVsTails')
broad_val_dir = os.path.join(validation_dir, 'HeadsVsTails')

# Data generators
broad_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
broad_train_gen = broad_datagen.flow_from_directory(
    broad_train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)
broad_val_gen = broad_datagen.flow_from_directory(
    broad_val_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# Broad category model
broad_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Heads and Tails
])

broad_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train broad model
broad_model.fit(
    broad_train_gen,
    steps_per_epoch=broad_train_gen.samples // batch_size,
    epochs=num_epochs,
    validation_data=broad_val_gen,
    validation_steps=broad_val_gen.samples // batch_size
)

# Save the trained model
broad_model.save('heads_vs_tails_model.h5')


def classify_and_split_images(model, source_dir, target_dir_heads, target_dir_tails, image_size):
    if not os.path.exists(target_dir_heads):
        os.makedirs(target_dir_heads)
    if not os.path.exists(target_dir_tails):
        os.makedirs(target_dir_tails)

    for subdir, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            img = tf.keras.preprocessing.image.load_img(file_path, target_size=(image_size, image_size))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict category
            prediction = model.predict(img_array)
            category = np.argmax(prediction)

            # Copy file to the appropriate folder
            if category == 0:  # Heads
                target_path = os.path.join(target_dir_heads, file)
            else:  # Tails
                target_path = os.path.join(target_dir_tails, file)
            
            shutil.copy(file_path, target_path)  # Copy file

# Split training data
classify_and_split_images(
    broad_model,
    source_dir=os.path.join(train_dir, 'All'),
    target_dir_heads=os.path.join(train_dir, 'Heads'),
    target_dir_tails=os.path.join(train_dir, 'Tails'),
    image_size=image_size
)

# Split validation data
classify_and_split_images(
    broad_model,
    source_dir=os.path.join(validation_dir, 'All'),
    target_dir_heads=os.path.join(validation_dir, 'Heads'),
    target_dir_tails=os.path.join(validation_dir, 'Tails'),
    image_size=image_size
)
'''
heads_train_dir = os.path.join(train_dir, 'Heads')
heads_val_dir = os.path.join(validation_dir, 'Heads')

fine_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
fine_train_gen = fine_datagen.flow_from_directory(
    heads_train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)
fine_val_gen = fine_datagen.flow_from_directory(
    heads_val_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

heads_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Quarters and Bicentennials
])

heads_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

heads_model.fit(
    fine_train_gen,
    steps_per_epoch=fine_train_gen.samples // batch_size,
    epochs=num_epochs,
    validation_data=fine_val_gen,
    validation_steps=fine_val_gen.samples // batch_size
)

heads_model.save('quarters_vs_bicentennials_heads_model.h5')

def evaluate_model(model, data_generator):
    predictions = np.argmax(model.predict(data_generator), axis=1)
    true_labels = data_generator.classes

    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(true_labels, predictions))

# Evaluate Heads vs Tails model
evaluate_model(broad_model, broad_val_gen)

# Evaluate Heads (Quarters vs Bicentennials)
evaluate_model(heads_model, fine_val_gen)

def predict_coin(image_path, broad_model, heads_model, tails_model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_size, image_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict broad category
    broad_prediction = broad_model.predict(img_array)
    broad_category = np.argmax(broad_prediction)

    if broad_category == 0:  # Heads
        fine_prediction = heads_model.predict(img_array)
        fine_category = np.argmax(fine_prediction)
        return "Heads -> " + ("Quarters" if fine_category == 0 else "Bicentennials")
    else:  # Tails
        fine_prediction = tails_model.predict(img_array)
        fine_category = np.argmax(fine_prediction)
        return "Tails -> " + ("Quarters" if fine_category == 0 else "Bicentennials")
'''