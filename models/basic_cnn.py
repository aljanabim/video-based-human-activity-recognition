"""Simple script preparing dat anad training a basic CNN.

IMPORTANT: Before running this, make sure the folder ./data/1000-videos/video exists and contains
           videos in .webm format. Also make sure that the foler
           ./data/20bn-something-something-v2-jason exists and contains four .json files.

"""

import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
sys.path.append('../')
sys.path.append('.')
from data_utils import dataset_builder
from data_utils import metadata_loader
from data_utils import video_to_frames
from config import Config


plt.style.use('ggplot')

# Config
config = Config(root_path='./data/1000-videos', img_width=84,
                img_height=84, use_subfolders=True)

# # Decode videos
# print("Start decode.")
# video_to_frames.decode_videos(config)
# print("Decode done.")

# Get metadata
ml = metadata_loader.MetadataLoader(config)
metadata = ml.load_metadata()

# Build datasets
db = dataset_builder.DatasetBuilder(config)
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
print("==== Train ====")
train_dataset = train_dataset.shuffle(buffer_size=100)
train_dataset = train_dataset.batch(100)
model.fit(train_dataset, epochs=1)

# Evaluate models
print("==== Evaluate ====")
valid_dataset = valid_dataset.batch(100)
model.evaluate(valid_dataset)
