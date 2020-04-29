import numpy as np
import tensorflow as tf


def extract_videos_and_labels(dict_dataset, n_classes=None):
    """Take dict dataset (train, test, or valid) and return lists of tensors and labels."""
    videos = []
    labels = []
    for video_id in dict_dataset:
        videos.append(dict_dataset[video_id]['data'])
        labels.append(dict_dataset[video_id]['action_label'])
    label_array = make_one_hot_encoding(labels, n_classes=n_classes)
    label_tensor = tf.convert_to_tensor(label_array)
    return videos, label_tensor


def extract_frames_and_labels(dict_dataset, n_classes=None):
    """Take dict dataset (train, test, or valid) and return lists of tensors and labels."""
    videos = []
    labels = []
    for video_id in dict_dataset:
        videos.append(dict_dataset[video_id]['data'])
        n_frames = dict_dataset[video_id]['data'].shape[0]
        for frame_number in range(n_frames):
            labels.append(dict_dataset[video_id]['action_label'])
    frames = tf.concat(videos, 0)

    label_array = make_one_hot_encoding(labels, n_classes=n_classes)
    label_tensor = tf.convert_to_tensor(label_array)
    return frames, label_tensor


def make_one_hot_encoding(labels, n_classes=None):
    """Take labels encoded as ints and return one-hot encoded array."""
    n_labels = len(labels)
    if n_classes is None:
        n_classes = max(labels)+1

    label_array = np.zeros((n_labels, n_classes))
    for i, label in enumerate(labels):
        label_array[i, label] = 1

    return label_array
