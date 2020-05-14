"""Basic script for training and evaluating a model on the kth actions dataset."""
import sys
sys.path.append('../')
sys.path.append('.')

import time
import numpy as np

# Imports
from data_utils import kth_dataset_builder
from config import Config
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

plt.style.use('ggplot')

IMG_WIDTH = 120
IMG_HEIGHT = 120
N_CLASSES = 6

def train_model():

    # Setup dataset builder
    video_path = './data/kth-actions/video'
    frame_path = './data/kth-actions/frame'
    builder = kth_dataset_builder.DatasetBuilder(
        video_path, frame_path, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, ms_per_frame=100, max_frames=16)

    # builder.convert_videos_to_frames()
    metadata = builder.generate_metadata()

    # Build frame datasets
    dataset_train = builder.make_frame_dataset(metadata=metadata['train']).batch(128).prefetch(1)
    dataset_valid = builder.make_frame_dataset(metadata=metadata['valid']).batch(128).prefetch(1)
    dataset_test = builder.make_frame_dataset(metadata=metadata['test']).batch(128).prefetch(1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(96,
                               11,
                               padding='same', activation='relu',
                               input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
       tf.keras.layers.Conv2D(48,
                              3,
                              padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(N_CLASSES)])


    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train model
    history = model.fit(dataset_train.take(100), epochs=10, validation_data=dataset_valid.take(10))

    # Evaluate models
    print("==== Evaluate ====")
    model.evaluate(dataset_valid)
    model.evaluate(dataset_test)

    model.save(".\\models\\trained_models\\mini_cnn")
    model.save_weights(".\\models\\checkpoints\\mini_cnn\\mini_cnn")

    # Plot training history
    plt.plot(history.history['accuracy'], label="Training accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation accuracy")
    plt.title("Accuracy during training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label="Training loss")
    plt.plot(history.history['val_loss'], label="Validation loss")
    plt.title("Loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    onehot_targets = [sample[1].numpy() for sample in dataset_valid.take(30)]
    onehot_targets = np.concatenate(onehot_targets, axis=0)
    targets = np.argmax(onehot_targets, axis=1)

    onehot_preds = model.predict(dataset_valid.take(30))
    preds = np.argmax(onehot_preds, axis=1)
    conf = confusion_matrix(targets, preds)

    ax = plt.subplot()
    sns.heatmap(conf, annot=True, ax=ax, cmap=sns.cubehelix_palette(8), fmt='g',
        xticklabels=['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'],
        yticklabels=['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    ax.set_title('Confusion Matrix')
    plt.show()

def load_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128,
                               2,
                               padding='same', activation='relu',
                               input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(56,
                               5,
                               padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(72,
                               5,
                               padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(56,
                               4,
                               padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D()])
    model.load_weights("./models/checkpoints/mini_cnn/mini_cnn")
    model.trainable = False
    model.summary()
    return model


if __name__ == '__main__':
    train_model()
