"""Module for building datasets that load procedurally from drive.

Example:
    # define paths
    root_path = ".\\data\\something-something-mini"
    anno_path = "{}-anno".format(root_path)
    frame_path = "{}-frame".format(root_path)

    # get metadata
    metadata_loader = MetadataLoader(label_folder_path=anno_path)
    metadata = metadata_loader.load_metadata()

    # get ids for videos in frame folder
    video_ids = os.listdir(frame_path)
    train_video_ids = [id for id in video_ids if int(id) in metadata['train']]
    video_id_list = train_video_ids

    builder = DatasetBuilder(max_frames=70,
                             n_classes=174,
                             img_width=455,
                             img_height=256,
                             frame_path=frame_path)

    video_dataset = builder.make_video_dataset(video_id_list, metadata['train'])
    frame_dataset = builder.make_frame_dataset(video_id_list, metadata['train'])

    print("=== VIDEOS ===")
    for video, label in video_dataset:
        print("shape:", video.shape, "label:", label.numpy())

    print("=== FRAMES ===")
    for frame, label in frame_dataset:
        print("shape:", frame.shape, "label:", label.numpy())

"""
import sys
sys.path.append('../')
sys.path.append('.')

from config import Config

from data_utils.metadata_loader import MetadataLoader
import tensorflow as tf
import os


class DatasetBuilder:
    """Used to build datasets."""

    def __init__(self, config):
        self.max_frames = config.max_frames
        self.n_classes = config.n_classes
        self.img_width = config.img_width
        self.img_height = config.img_height
        self.frame_path = config.frame_path
        self.autotune = tf.data.experimental.AUTOTUNE

    def _build_process_path_function(self, action_label_table, img_width, img_height, n_classes):

        def _make_one_hot_encoding(label_integer):
            """Take labels encoded as ints and return one-hot encoded array."""
            label_tensor = tf.one_hot(label_integer, n_classes, dtype=tf.int32)
            return label_tensor

        def _get_label(file_path):
            # convert the path to a list of path components

            # special case for windows file systems
            if os.name == 'nt':  # nt == windows
                parts = tf.strings.split(file_path, sep="\\")
            else:
                parts = tf.strings.split(file_path, sep="/")
            # The second to last is the class-directory
            return action_label_table.lookup(parts[-2])

        def _decode_image(frame):  # from einar's script
            frame = tf.image.decode_jpeg(frame, channels=3)
            frame = tf.image.convert_image_dtype(frame, tf.float32)
            return tf.image.resize(frame, [img_width, img_height])

        def _process_path(file_path):
            # ------------------------------------------------
            # parts = tf.strings.split(file_path, sep="\\")
            # label = action_label_table.lookup(parts[-2])
            # tf.print(label)
            # label_tensor = tf.one_hot(label, n_classes, dtype=tf.int32)
            # img = tf.io.read_file(file_path)
            # img = tf.image.decode_jpeg(img, channels=3)
            # img = tf.image.convert_image_dtype(img, tf.float32)
            # frame = tf.image.resize(img, [img_width, img_height])

            label = _get_label(file_path)
            label_tensor = _make_one_hot_encoding(label)
            img = tf.io.read_file(file_path)  # from einar's script
            img = _decode_image(img)
            return img, label_tensor

        return _process_path

    def _build_stack_images_from_path_function(self, process_path_function):

        def _stack_images_from_path(ds):
            labeled_ds = ds.map(process_path_function,
                                num_parallel_calls=self.autotune)

            # temp variables
            label = tf.constant(0, dtype=tf.int32)
            i = tf.constant(0)
            imgs_combined = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True,
                                           clear_after_read=False)

            for _, first_label in labeled_ds.take(1):
                label = first_label

            for im, labels in labeled_ds:
                imgs_combined = imgs_combined.write(i, im)
                i = tf.add(i, 1)

            return imgs_combined.stack(), label

        return _stack_images_from_path

    def _dataset_from_folder(self, file):
        return tf.data.Dataset.list_files(file + "/*", shuffle=False)

    def _build_pad_function(self, max_frames):

        def _pad(stacked_im, label):
            nr = max_frames - stacked_im.get_shape().as_list()[0]

            paddings = tf.constant([[0, nr], [0, 0], [0, 0], [0, 0]])
            new = tf.pad(stacked_im, paddings, "CONSTANT")
            return new, label

        def _pad_fn(stacked_im, label):
            padded_im, label = tf.py_function(_pad,
                                              inp=[stacked_im, label],
                                              Tout=(tf.float32, tf.int32))
            return padded_im, label

        return _pad_fn

    def make_video_dataset(self, metadata):
        """Take list of frame folder paths and return dataset with videos."""

        video_id_list = [id for id in os.listdir(
            self.frame_path) if int(id) in metadata]
        # creates a dataset containing frame folder paths
        frame_folder_paths = [self.frame_path +
                              "/" + id for id in video_id_list]
        frame_folder_paths_dataset = tf.data.Dataset.from_tensor_slices(
            frame_folder_paths)

        # creates dataset containing datasets of frame paths
        frame_folder_dataset = frame_folder_paths_dataset.map(
            self._dataset_from_folder, num_parallel_calls=self.autotune)

        # creates list of labels
        action_labels = [metadata[int(id)]['action_label']
                         for id in video_id_list]
        action_label_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(video_id_list, action_labels), -1)

        # build functions to process images and apply map
        process_path_function = self._build_process_path_function(
            action_label_table, self.img_width, self.img_height, n_classes=self.n_classes)
        stack_images_from_path_function = self._build_stack_images_from_path_function(
            process_path_function)
        video_dataset = frame_folder_dataset.map(
            stack_images_from_path_function, num_parallel_calls=self.autotune)

        # build padding function and apply
        pad_function = self._build_pad_function(self.max_frames)
        padded_videos_dataset = video_dataset.map(pad_function)

        return padded_videos_dataset

    def make_frame_dataset(self, metadata):
        """Take list of frame folder paths and return dataset with frames."""
        video_id_list = [id for id in os.listdir(
            self.frame_path) if int(id) in metadata]
        frame_folder_paths = [self.frame_path +
                              "/" + id for id in video_id_list]

        # creates a dataset containing frame folder paths
        frame_path_subdatasets = [self._dataset_from_folder(
            path) for path in frame_folder_paths]

        if len(frame_path_subdatasets) == 0:
            return

        frame_path_dataset = frame_path_subdatasets.pop()
        while frame_path_subdatasets:
            frame_path_dataset = frame_path_dataset.concatenate(
                frame_path_subdatasets.pop())

        # for path in frame_path_dataset:
        #     print(path)

        # creates list of labels
        action_labels = [metadata[int(id)]['action_label']
                         for id in video_id_list]
        action_label_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(video_id_list, action_labels), -1)

        process_path_function = self._build_process_path_function(
            action_label_table, self.img_width, self.img_height, n_classes=self.n_classes)

        frame_dataset = frame_path_dataset.map(
            process_path_function, num_parallel_calls=self.autotune)

        return frame_dataset


if __name__ == '__main__':
    # load config
    config = Config()

    # get metadata
    metadata_loader = MetadataLoader(config)
    metadata = metadata_loader.load_metadata()

    # get ids for videos in frame folder
    video_ids = os.listdir(config.frame_path)
    train_video_ids = [id for id in video_ids if int(id) in metadata['train']]
    video_id_list = train_video_ids

    builder = DatasetBuilder(config)

    video_dataset = builder.make_video_dataset(metadata['train'])
    frame_dataset = builder.make_frame_dataset(metadata['train'])

    print("=== VIDEOS ===")
    for video, label in video_dataset:
        print("shape:", video.shape, "label:", label.numpy())

    print("=== FRAMES ===")
    for frame, label in frame_dataset.take(1):
        print(label.numpy())
        print("shape:", frame.shape, "label:", label.numpy())
