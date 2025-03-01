import os
import zipfile
import tensorflow as tf
#from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

local_zip = '/tmp/coins/qtest2.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/coins')
zip_ref.close()
image_size = 150
num_epochs = 50
base_dir = '/tmp/coins/qtest2'
train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'Testing')

# # Directory with our training cat pictures
# train_cats_dir = os.path.join(train_dir, 'Quarters')

# # Directory with our training dog pictures
# train_dogs_dir = os.path.join(train_dir, 'Bicentennials')

# # Directory with our validation cat pictures
# validation_cats_dir = os.path.join(validation_dir, 'Quarters')

# # Directory with our validation dog pictures
# validation_dogs_dir = os.path.join(validation_dir, 'Bicentennials')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(image_size, image_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
            #   optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              optimizer=tf.keras.optimizers.Adagrad(),
              metrics=['accuracy'])

# This code has changed. Now instead of the ImageGenerator just rescaling
# the image, we also rotate and do other operations
# Updated to do image augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255,
      rotation_range=180,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=False,
      fill_mode='nearest')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(image_size, image_size),  # All images will be resized to 150x150
        batch_size=164,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=28,
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=2,  # 253 images = batch_size * steps
      epochs=15,       #100 epochs to 50 epochs
      validation_data=validation_generator,
      validation_steps=1,  # 42 images = batch_size * steps
      verbose=2)

#updated batch size from 20 to 10
#updated steps per epoch from 100 to 25 (training) and 50 to 4 (testing)


# predicting images
import numpy as np
img = tf.keras.utils.load_img("/tmp/coins/predict/coin1.png", target_size=(image_size, image_size))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0]>0.5:
    print(" bicentenial")
else:
    print(" quarter")


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()