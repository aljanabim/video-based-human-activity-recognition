from data_utils.data_loader import load_data
from data_utils import preprocess
import tensorflow as tf

N_CLASSES = 174

data, label_dict = load_data("./data/something-something-mini")

videos, labels = preprocess.extract_videos_and_labels(data['train'], n_classes=N_CLASSES)
frames, labels = preprocess.extract_frames_and_labels(data['train'], n_classes=N_CLASSES)
print(labels.shape)
# videos_dataset = tf.data.Dataset.from_tensors(videos) # currently broken
frames_dataset = tf.data.Dataset.from_tensor_slices((frames, labels))
print(frames_dataset)
