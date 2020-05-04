from data_utils import data_loader
from data_utils import preprocess
from data_utils import records
import tensorflow as tf
from config import Config
import os

config = Config()

# Load data from disk
data, label_dict = data_loader.load_data(config)

# Extract data and labels
videos, labels = preprocess.extract_videos_and_labels(data['train'], n_classes=config.n_classes)
frames, labels = preprocess.extract_frames_and_labels(data['train'], n_classes=config.n_classes)

# Find relevant dimensions
example_frame = frames[0]
example_label = labels[0]
frame_height = example_frame.shape[0]
frame_width = example_frame.shape[1]
frame_n_channels = example_frame.shape[2]
n_classes = example_label.shape[0]

# Build datasets
videos_dataset = tf.data.Dataset.from_generator(
    generator=lambda: ((video, label) for video, label in zip(videos, labels)),
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, frame_height, frame_width, frame_n_channels), (n_classes,)))
frames_dataset = tf.data.Dataset.from_tensor_slices((frames, labels))

# Make tfrecords
if not os.path.exists(config.record_output):
    os.makedirs(config.record_output)
records.serialize_dataset(dataset=frames_dataset,
                          output_directory=config.record_output+'record',
                          name="test-1")
