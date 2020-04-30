from data_utils.data_loader import load_data
from data_utils import preprocess
import tensorflow as tf

N_CLASSES = 174

# Load data from disk
data, label_dict = load_data("./data/something-something-mini")

# Extract data and labels
videos, labels = preprocess.extract_videos_and_labels(data['train'], n_classes=N_CLASSES)
frames, labels = preprocess.extract_frames_and_labels(data['train'], n_classes=N_CLASSES)

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

print(videos_dataset)
print(frames_dataset)

# Make tfrecords
# TODO: make tf records
