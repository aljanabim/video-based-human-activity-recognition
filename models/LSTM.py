# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import sys
sys.path.append('../')
sys.path.append('.')

from config import Config
from data_utils import video_to_frames
from data_utils import metadata_loader
from data_utils import dataset_builder

# %%
config = Config(root_path='../data/1000-videos', img_width=84,
                img_height=84, use_subfolders=True)
# config = Config()

# Get metadata
ml = metadata_loader.MetadataLoader(config)
metadata = ml.load_metadata()

# Build datasets
db = dataset_builder.DatasetBuilder(config)
train_dataset = db.make_video_dataset(metadata['train'])
valid_dataset = db.make_video_dataset(metadata['valid'])
test_dataset = db.make_video_dataset(metadata['test'])
# %% Resize Images

IMG_SIZE = 160  # All images will be resized to 160x160


def format_example(image, label):
    #image = tf.cast(image, tf.float32)
    image.set_shape((None, config.img_height, config.img_height, 3))
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    label.set_shape([config.n_classes])
    return image, label


train = train_dataset.map(format_example)
validation = valid_dataset.map(format_example)
test = test_dataset.map(format_example)

# %% Set batchsize and shuffle
BATCH_SIZE = 8
SHUFFLE_BUFFER_SIZE = 1
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
    pass
print(image_batch.shape, label_batch.shape)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

# layers
# def get_model(base_model):
# Layers
global_pooling = tf.keras.layers.GlobalAveragePooling2D()
feature_extractor = tf.keras.Sequential([base_model, global_pooling])

features = tf.keras.layers.TimeDistributed(
    feature_extractor, input_shape=(None, 160, 160, 3))


lstm = tf.keras.layers.LSTM(60, input_shape=(None, 1280))
lstm.build((None, None, 1280))

dense1 = tf.keras.layers.Dense(512, kernel_initializer="he_normal")

batch_normalization = tf.keras.layers.BatchNormalization()

classifier_dense = tf.keras.layers.Dense(174)


# model
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    full_model = tf.keras.Sequential([
        features,
        lstm,
        dense1,
        batch_normalization,
        classifier_dense])
    base_learning_rate = 0.0001
    full_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                       loss=tf.keras.losses.CategoricalCrossentropy(
                           from_logits=True),
                       metrics=['accuracy'])

# full_model = tf.keras.Sequential([
#     features,
#     lstm,
#     dense1,
#     batch_normalization,
#     classifier_dense])

# compile model

    # return full_model


# full_model = get_model(base_model)
full_model.summary()
# print(full_model(image_batch).shape)

# %% Train model
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
full_model.fit(train_batches, epochs=14)
