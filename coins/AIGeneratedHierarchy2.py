import os
import shutil
import numpy as np
import tensorflow as tf
#from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from Categorize import findHierarchyMiscategorizedImages

# --- PARAMETERS ---
image_size = 150
batch_size = 3
num_epochs = 30

base_dir = '/home/noah/projects/image-identifier/data/coins/BalancedAIHierarchy'
train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'Testing')
misclassificationFolder = os.path.join(base_dir, 'Misclassified')

# Paths for broad category
broad_train_dir = os.path.join(train_dir, 'HeadsVsTails')
broad_val_dir = os.path.join(validation_dir, 'HeadsVsTails')

# Paths for fine-grained categories
heads_train_dir = os.path.join(train_dir, 'Heads')
tails_train_dir = os.path.join(train_dir, 'Tails')
heads_val_dir = os.path.join(validation_dir, 'Heads')
tails_val_dir = os.path.join(validation_dir, 'Tails')

# --- HELPER FUNCTIONS ---
def classify_and_split_images(model, source_dir, target_dir_heads, target_dir_tails, image_size):
    """Classify images into Heads or Tails and copy them to their respective folders."""
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

            shutil.copy(file_path, target_path)

def evaluate_model(model, data_generator):
    """Evaluate a model using confusion matrix and classification report."""
    predictions = np.argmax(model.predict(data_generator), axis=1)
    true_labels = data_generator.classes

    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(true_labels, predictions))

def plot_training(history, category_name):
    """Plot training and validation accuracy/loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label=f'{category_name} Training Acc')
    plt.plot(epochs, val_acc, 'r', label=f'{category_name} Validation Acc')
    plt.title(f'{category_name} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label=f'{category_name} Training Loss')
    plt.plot(epochs, val_loss, 'r', label=f'{category_name} Validation Loss')
    plt.title(f'{category_name} Loss')
    plt.legend()

    plt.show()

# --- STEP 1: TRAIN HEADS VS TAILS MODEL ---
print("Training Heads vs Tails model...")
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

broad_history = broad_model.fit(
    broad_train_gen,
    steps_per_epoch=broad_train_gen.samples // batch_size,
    epochs=num_epochs,
    validation_data=broad_val_gen,
    validation_steps=broad_val_gen.samples // batch_size
)

broad_model.save('heads_vs_tails_model.h5')
plot_training(broad_history, "Heads vs Tails")

# --- STEP 2: SPLIT IMAGES INTO HEADS AND TAILS ---
# print("Routing images using Heads vs Tails model...")
# classify_and_split_images(
#     broad_model,
#     source_dir=os.path.join(train_dir, 'All'),
#     target_dir_heads=heads_train_dir,
#     target_dir_tails=tails_train_dir,
#     image_size=image_size
# )
# classify_and_split_images(
#     broad_model,
#     source_dir=os.path.join(validation_dir, 'All'),
#     target_dir_heads=heads_val_dir,
#     target_dir_tails=tails_val_dir,
#     image_size=image_size
# )

# --- STEP 3: TRAIN FINE-GRAINED MODELS ---
print("Training Quarters vs Bicentennials (Heads)...")
heads_train_gen = broad_datagen.flow_from_directory(
    heads_train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)
heads_val_gen = broad_datagen.flow_from_directory(
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

heads_history = heads_model.fit(
    heads_train_gen,
    steps_per_epoch=heads_train_gen.samples // batch_size,
    epochs=num_epochs,
    validation_data=heads_val_gen,
    validation_steps=heads_val_gen.samples // batch_size
)

heads_model.save('quarters_vs_bicentennials_heads_model.h5')
plot_training(heads_history, "Quarters vs Bicentennials (Heads)")

print("Training Quarters vs Bicentennials (Tails)...")
tails_train_gen = broad_datagen.flow_from_directory(
    tails_train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)
tails_val_gen = broad_datagen.flow_from_directory(
    tails_val_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

tails_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Quarters and Bicentennials
])

tails_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

tails_history = tails_model.fit(
    tails_train_gen,
    steps_per_epoch=tails_train_gen.samples // batch_size,
    epochs=num_epochs,
    validation_data=tails_val_gen,
    validation_steps=tails_val_gen.samples // batch_size
)

tails_model.save('quarters_vs_bicentennials_tails_model.h5')
plot_training(tails_history, "Quarters vs Bicentennials (Tails)")

# --- STEP 4: EVALUATE MODELS ---
print("Evaluating Heads vs Tails model...")
evaluate_model(broad_model, broad_val_gen)

print("Evaluating Quarters vs Bicentennials (Heads)...")
evaluate_model(heads_model, heads_val_gen)

print("Evaluating Quarters vs Bicentennials (Tails)...")
evaluate_model(tails_model, tails_val_gen)

# #Step 5: Inference Workflow
# def predict_coin(image_path, broad_model, heads_model, tails_model):
#     """Predict the category of a coin image using the hierarchical model."""
#     # Preprocess image
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_size, image_size))
#     img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Step 1: Predict Heads vs Tails
#     broad_prediction = broad_model.predict(img_array)
#     broad_category = np.argmax(broad_prediction)

#     if broad_category == 0:  # Heads
#         fine_prediction = heads_model.predict(img_array)
#         fine_category = np.argmax(fine_prediction)
#         return "Heads -> " + ("Quarters" if fine_category == 0 else "Bicentennials")
#     else:  # Tails
#         fine_prediction = tails_model.predict(img_array)
#         fine_category = np.argmax(fine_prediction)
#         return "Tails -> " + ("Quarters" if fine_category == 0 else "Bicentennials")

# Example usage
# new_image_path = '/path/to/new/image.jpg'
# result = predict_coin(new_image_path, broad_model, heads_model, tails_model)
# print("Predicted category:", result)


findHierarchyMiscategorizedImages(broad_model, heads_model, tails_model, validation_dir, misclassificationFolder, image_size)