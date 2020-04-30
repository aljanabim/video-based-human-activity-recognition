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

def _tensor_to_bytes_feature(value):
    if tf.executing_eagerly():
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy().tostring()]))
    else:
        print(value)
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy.tostring()]))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _serialize_frame_sample(frame, label):
    # print(frame)
    feature = {'frame': _tensor_to_bytes_feature(frame),
               'label': _tensor_to_bytes_feature(label)}
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def tf_serialize_frame_sample(frame, label):
    tf_string = tf.py_function(
        _serialize_frame_sample,
        (frame, label),
        tf.string)
    return tf.reshape(tf_string, ())


for frame, label in frames_dataset.take(1):
    # print(frame.numpy())
    _serialize_frame_sample(frame, label)

print("Starting serialization")
serialized_frames_dataset = frames_dataset.map(tf_serialize_frame_sample)
print("Serialization complete")
record_file_path = "test.tfrecord"
writer = tf.data.experimental.TFRecordWriter(record_file_path)
writer.write(serialized_frames_dataset)
print("TFRecord created")
