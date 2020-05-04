from data_utils import video_to_frames
from data_utils import metadata_loader
from data_utils import dataset_builder
import os
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('ggplot')

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
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                           input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(n_classes)])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Train model
train_dataset = train_dataset.shuffle(buffer_size=100)
train_dataset = train_dataset.batch(20)
model.fit(train_dataset, epochs=2)
