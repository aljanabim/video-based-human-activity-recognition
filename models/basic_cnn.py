from data_utils import video_to_frames
from data_utils import metadata_loader
from data_utils import dataset_builder
import os

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import tensorflow as tf

# Data hyperparams
max_frames = 80
n_classes = 174
img_width = 84
img_height = 84

# Decode videos
videos_path = './data/something-something-mini-video'
json_path = './data/something-something-mini-anno'
output_path = './data/testdata'
video_to_frames.decode_videos(videos_path, json_path, output_path,
                              img_width=img_width, img_height=img_height)

# Get metadata
frames_path = output_path + "/frames"
labels_path = output_path + "/labels"
ml = metadata_loader.MetadataLoader(label_folder_path=labels_path)
metadata = ml.load_metadata()

# Get video id sets
video_ids = os.listdir(frames_path)
train_video_ids = [id for id in video_ids if int(id) in metadata['train']]
valid_video_ids = [id for id in video_ids if int(id) in metadata['valid']]
test_video_ids = [id for id in video_ids if int(id) in metadata['test']]

# Setup dataset builder
db = dataset_builder.DatasetBuilder(max_frames=max_frames,
                                    n_classes=n_classes,
                                    img_width=img_width,
                                    img_height=img_height,
                                    frame_path=frames_path)

# Build datasets
train_dataset = db.make_frame_dataset(train_video_ids, metadata['train'])
valid_dataset = db.make_frame_dataset(valid_video_ids, metadata['valid'])
test_dataset = db.make_frame_dataset(test_video_ids, metadata['test'])

# Build model

for sample in train_dataset.take(1):
    print(sample[1].numpy())

model = tf.keras.Sequential([
    tf.keras.Conv2D(16, 3, paddin='same', activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.Flatten(),
    tf.keras.Dense(512, activation='relu'),
    Dense
])
