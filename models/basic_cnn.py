import sys
sys.path.append('../')
sys.path.append('.')

from config import Config
from data_utils import video_to_frames
from data_utils import metadata_loader
from data_utils import dataset_builder
import os
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# hyperparams
config = Config()

# Decode videos
video_to_frames.decode_videos(config)

# Get metadata
ml = metadata_loader.MetadataLoader(config)
metadata = ml.load_metadata()


# Setup dataset builder
db = dataset_builder.DatasetBuilder(config)

# Build datasets
train_dataset = db.make_frame_dataset(metadata['train'])
valid_dataset = db.make_frame_dataset(metadata['valid'])
test_dataset = db.make_frame_dataset(metadata['test'])

# Build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                           input_shape=(config.img_height, config.img_width, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(config.n_classes)])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Train model
train_dataset = train_dataset.shuffle(buffer_size=100)
train_dataset = train_dataset.batch(20)
model.fit(train_dataset, epochs=2)
