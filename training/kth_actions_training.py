"""Basic script for training and evaluating a model on the kth actions dataset."""
import sys
sys.path.append('../')
sys.path.append('.')

import time

# Imports
from data_utils import kth_dataset_builder
from config import Config
import tensorflow as tf
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

plt.style.use('ggplot')

IMG_WIDTH = 84
IMG_HEIGHT = 84
N_CLASSES = 6

# Setup dataset builder
video_path = './data/kth-actions/video'
frame_path = './data/kth-actions/frame'
builder = kth_dataset_builder.DatasetBuilder(
    video_path, frame_path, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, ms_per_frame=100, max_frames=20)

# builder.convert_videos_to_frames()
metadata = builder.generate_metadata()

# Build frame datasets
dataset_train = builder.make_frame_dataset(metadata=metadata['train']).take(2560).shuffle(100).batch(128).prefetch(1)
dataset_valid = builder.make_frame_dataset(metadata=metadata['valid']).take(1280).batch(128).prefetch(1)
dataset_test = builder.make_frame_dataset(metadata=metadata['test']).take(1280).batch(128).prefetch(1)

start = time.time()
for sample in dataset_train:
    print(sample[0].shape)
print("First iteration time: {}".format(time.time() - start))

start = time.time()
for sample in dataset_train:
    print(sample[0].shape)
print("Second iteration time: {}".format(time.time() - start))

#
# # Build CNN model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
#                            input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu',
#                            input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(N_CLASSES)])
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# model.summary()
#
# # Train model
# history = model.fit(dataset_train, epochs=5, validation_data=dataset_valid)
#
# # Evaluate models
# print("==== Evaluate ====")
# model.evaluate(dataset_valid)
#
# # Plot training history
#
# plt.plot(history.history['accuracy'], label="Training accuracy")
# plt.plot(history.history['val_accuracy'], label="Validation accuracy")
# plt.title("Accuracy during training")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()
#
# plt.plot(history.history['loss'], label="Training loss")
# plt.plot(history.history['val_loss'], label="Validation loss")
# plt.title("Loss during training")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
