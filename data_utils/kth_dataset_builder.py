""" Module for processing KTH Actions dataset.

Instructions:

1. Download dataset from https://www.csc.kth.se/cvap/actions/
2. Create following folder structure in project directory:
    .
    └── data
        └── kth-actions
            ├── frame
            └── video
                ├── boxing
                ├── handclapping
                ├── handwaving
                ├── jogging
                ├── running
                └── walking

3. Extract each class into corresponding video folder.
4. Run following script:

    video_path = './data/kth-actions/video'
    frame_path = './data/kth-actions/frame'

    builder = DatasetBuilder(video_path, frame_path, img_width=84, img_height=84, ms_per_frame=1000)
    builder.convert_videos_to_frames()
    metadata = builder.generate_metadata()

    video_dataset_train = builder.make_video_dataset(metadata=metadata['train'])
    video_dataset_valid = builder.make_video_dataset(metadata=metadata['valid'])
    video_dataset_test = builder.make_video_dataset(metadata=metadata['test'])

    frame_dataset_train = builder.make_frame_dataset(metadata=metadata['train'])
    frame_dataset_valid = builder.make_frame_dataset(metadata=metadata['valid'])
    frame_dataset_test = builder.make_frame_dataset(metadata=metadata['test'])

5. Train neural networks and make big money

"""
import cv2
import os
import tensorflow as tf
import random
import random
import numpy as np


class DatasetBuilder:
    """Converts videos to jpg and builds datasets."""

    def __init__(self, video_path, frame_path, img_width=84, img_height=84,
                 ms_per_frame=1000, max_frames=25):
        """Init DatasetBuilder.

        The following folder structure should exist:

        .
        └── data
            └── dataset_name
                ├── frame
                └── video
                    ├── boxing
                    ├── handclapping
                    ├── handwaving
                    ├── jogging
                    ├── running
                    └── walking

        In this case, video_path = ./data/dataset_name/video
                      frame_path = ./data/dataset_name/frame

        """
        self.video_path = video_path
        self.frame_path = frame_path
        self.ms_per_frame = ms_per_frame
        self.img_width = img_width
        self.img_height = img_height
        self.img_shape = (img_width, img_height)
        self.n_videos = 600
        self.max_frames = max_frames

        self.n_classes = 6
        self.label_names = os.listdir(self.video_path)
        self.label_ints = [0, 1, 2, 3, 4, 5]

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
            return action_label_table.lookup(parts[-3])

        def _decode_image(frame):  # from einar's script
            frame = tf.image.decode_jpeg(frame, channels=1)
            frame = tf.image.convert_image_dtype(frame, tf.float32)
            return tf.image.resize(frame, [img_width, img_height])

        def _process_path(file_path):
            label = _get_label(file_path)
            label_tensor = _make_one_hot_encoding(label)
            img = tf.io.read_file(file_path)  # from einar's script
            img = _decode_image(img)
            return img, label_tensor

        return _process_path

    def _build_stack_images_from_path_function(self, process_path_function):

        def _stack_images_from_path(ds):
            labeled_ds = ds.map(process_path_function, num_parallel_calls=self.autotune)

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
        return tf.data.Dataset.list_files(file+"/*", shuffle=False)

    def _slice_from_folder(self, file):
        slice = tf.py_function(self._py_slice_from_folder, [file], tf.string)

        return tf.data.Dataset.list_files(slice)

    def _py_slice_from_folder(self, file):
        frames = tf.io.gfile.listdir(file.numpy())
        frames = [file + '/' + frame for frame in frames]
        n_frames = len(frames)
        if n_frames <= self.max_frames:
            slice = frames
        else:
            r = random.randint(0, n_frames-self.max_frames)
            slice = frames[r:r + self.max_frames]

        return tf.convert_to_tensor(slice)

    def _build_pad_function(self, max_frames):

        def _pad(stacked_im, label):
            nr = max_frames - stacked_im.get_shape().as_list()[0]

            paddings = tf.constant([[0, nr], [0, 0], [0,0], [0,0]])
            new = tf.pad(stacked_im, paddings, "CONSTANT")
            return new, label

        def _pad_fn(stacked_im, label):
            padded_im, label = tf.py_function(_pad,
                                              inp=[stacked_im, label],
                                              Tout=(tf.float32, tf.int32))
            return padded_im, label

        return _pad_fn

    def make_video_dataset(self, metadata):
        """Take metadata dict and return dataset with videos."""
        # get relevant folder paths
        frame_folder_paths = []
        for label_name in self.label_names:
            video_names = os.listdir(self.frame_path + '/' + label_name)
            video_names = [name for name in video_names if name in metadata]  # Filter out set
            paths = ["{}/{}/{}".format(self.frame_path, label_name, video_name) for video_name in video_names]
            frame_folder_paths.extend(paths)

        random.shuffle(frame_folder_paths)

        # create nested dataset
        frame_folder_paths_dataset = tf.data.Dataset.from_tensor_slices(frame_folder_paths)
        frame_folder_dataset = frame_folder_paths_dataset.map(
            self._slice_from_folder, num_parallel_calls=self.autotune)

        # create label table
        action_label_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.label_names, self.label_ints), -1)

        # build functions to process images into stacked tensors with labels and apply map
        process_path_function = self._build_process_path_function(
            action_label_table, self.img_width, self.img_height, n_classes=self.n_classes)
        stack_images_from_path_function = self._build_stack_images_from_path_function(
            process_path_function)
        video_dataset = frame_folder_dataset.map(
            stack_images_from_path_function, num_parallel_calls=self.autotune)

        # build padding function and apply
        pad_function = self._build_pad_function(self.max_frames)
        padded_videos_dataset = video_dataset.map(pad_function, num_parallel_calls=self.autotune)


        # Set shapes
        def format_example(image, label):
            image.set_shape((None, self.img_width, self.img_height, 1))
            label.set_shape([self.n_classes])
            return image, label

        padded_videos_dataset = padded_videos_dataset.map(format_example, num_parallel_calls=self.autotune)

        return padded_videos_dataset.shuffle(1000)

    def make_frame_dataset(self, metadata):
        """Take metadata dict and return dataset with frames."""
        # get relevant folder paths
        frame_folder_paths = []
        for label_name in self.label_names:
            video_names = os.listdir(self.frame_path + '/' + label_name)
            video_names = [name for name in video_names if name in metadata]  # Filter out set
            paths = ["{}/{}/{}".format(self.frame_path, label_name, video_name) for video_name in video_names]
            frame_folder_paths.extend(paths)

        frame_paths = []
        for folder in frame_folder_paths:
            frames = os.listdir(folder)
            frame_paths.extend([folder + '/' + frame for frame in frames])

        random.shuffle(frame_paths)

        # concatenate frame paths datasets
        frame_path_dataset = tf.data.Dataset.list_files(frame_paths)

        # creates list of labels
        action_label_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.label_names, self.label_ints), -1)

        # build function to process images into tensors with labels and apply map
        process_path_function = self._build_process_path_function(
            action_label_table, self.img_width, self.img_height, n_classes=self.n_classes)
        frame_dataset = frame_path_dataset.map(process_path_function, num_parallel_calls=self.autotune)

        return frame_dataset.shuffle(2000, reshuffle_each_iteration=True)

    def convert_videos_to_frames(self):
        """Convert videos on disk to jpg frames and store on disk.

        Must have below folder structure:

            dataset_name
            ├── frame
            └── video
                ├── boxing
                ├── handclapping
                ├── handwaving
                ├── jogging
                ├── running
                └── walking

        where the video folders contains the corresponding videos. Any existing frames in the frame
        folders will be overwritten.

        """
        print("Converting videos... 0% done")
        counter = 0
        classes = os.listdir(self.video_path)

        for classname in classes:
            if os.path.exists(self.frame_path + '/' + classname): continue
            os.mkdir(self.frame_path + '/' + classname)
            files = os.listdir("{}/{}".format(self.video_path, classname))
            file_paths = ["{}/{}/{}".format(self.video_path, classname, filename) for filename in files]
            for path in file_paths:
                self._video_to_frames(path)
                counter += 1
                if counter % 60 == 0:
                    print("Converting videos... {}% done".format(int((counter/self.n_videos)*100)))
        print("Conversion done, frames stored in {}".format(self.frame_path))

    def _video_to_frames(self, video_path):
        video_path.replace('\\', '/')
        parts = video_path.split('/')
        classname = parts[-2]
        videoname = parts[-1].strip('.avi')

        output_dir = "{}/{}/{}".format(self.frame_path, classname, videoname)

        try:
            os.mkdir(output_dir)
        except FileExistsError:
            pass

        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * self.ms_per_frame))
            framename = "{}_{}".format(videoname, count)
            output_path = r"{}/{}.jpg".format(output_dir, framename)
            image = cv2.resize(image, self.img_shape)
            cv2.imwrite(output_path, image)
            success, image = vidcap.read()
            count += 1
        return

    def generate_metadata(self):
        """Return a nested metadata dict.

        Should be used after convert_videos_to_frames().
        """
        valid_frac = 0.15
        test_frac = 0.15
        metadata = {}
        metadata['train'] = {}
        metadata['valid'] = {}
        metadata['test'] = {}
        for label_name in self.label_names:
            video_names = os.listdir(self.frame_path + '/' + label_name)
            n_videos = len(video_names)
            valid_start = -int(n_videos*(valid_frac + test_frac))
            test_start = -int(n_videos*(test_frac))
            train_videos = video_names[:valid_start]
            valid_videos = video_names[valid_start:test_start]
            test_videos = video_names[test_start:]
            metadata['train'].update({id: label_name for id in train_videos})
            metadata['valid'].update({id: label_name for id in valid_videos})
            metadata['test'].update({id: label_name for id in test_videos})
        return metadata


if __name__ == "__main__":
    # Setup builder
    video_path = './data/kth-actions/video'
    frame_path = './data/kth-actions/frame'
    builder = DatasetBuilder(video_path, frame_path, img_width=84, img_height=84, ms_per_frame=100,
                             max_frames=50)

    # Convert videos and generate metadata
    builder.convert_videos_to_frames()
    metadata = builder.generate_metadata()

    # Build datasets
    video_dataset_train = builder.make_video_dataset(metadata=metadata['train'])
    video_dataset_valid = builder.make_video_dataset(metadata=metadata['valid'])
    video_dataset_test = builder.make_video_dataset(metadata=metadata['test'])

    frame_dataset_train = builder.make_frame_dataset(metadata=metadata['train']).take(10)
    # frame_dataset_valid = builder.make_frame_dataset(metadata=metadata['valid'])
    # frame_dataset_test = builder.make_frame_dataset(metadata=metadata['test'])

    # Verify that the datasets work
    LABELS = os.listdir(video_path)

    print("THE VIDEO DATASET")
    for vid, label in video_dataset_train:
        print(vid.shape, LABELS[np.argmax(label.numpy())])

    print("THE FRAME DATASET")
    for frame, label in frame_dataset_train:
        print(frame.shape, LABELS[np.argmax(label.numpy())])
