import cv2
import os
import tensorflow as tf

class DatasetBuilder:

    def __init__(self, video_path, frame_path, img_width, img_height, ms_per_frame=1000, max_frames=20):
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
        """Take list of frame folder paths and return dataset with videos."""

        frame_folder_paths = []
        for label_name in self.label_names:
            video_names = os.listdir(self.frame_path + '/' + label_name)
            paths = ["{}/{}/{}".format(self.frame_path, label_name, video_name) for video_name in video_names]
            frame_folder_paths.extend(paths)

        frame_folder_paths_dataset = tf.data.Dataset.from_tensor_slices(frame_folder_paths)

        # creates dataset containing datasets of frame paths
        frame_folder_dataset = frame_folder_paths_dataset.map(
            self._dataset_from_folder, num_parallel_calls=self.autotune)


        # creates list of labels

        action_label_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.label_names, self.label_ints), -1)

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
        """Take list of frame folder paths and return dataset with videos."""

        frame_folder_paths = []
        for label_name in self.label_names:
            video_names = os.listdir(self.frame_path + '/' + label_name)
            paths = ["{}/{}/{}".format(self.frame_path, label_name, video_name) for video_name in video_names]
            frame_folder_paths.extend(paths)

        frame_path_subdatasets = [self._dataset_from_folder(path) for path in frame_folder_paths]

        frame_path_dataset = frame_path_subdatasets.pop()
        while frame_path_subdatasets:
            frame_path_dataset = frame_path_dataset.concatenate(frame_path_subdatasets.pop())

        # creates list of labels

        action_label_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self.label_names, self.label_ints), -1)

        process_path_function = self._build_process_path_function(
            action_label_table, self.img_width, self.img_height, n_classes=self.n_classes)

        frame_dataset = frame_path_dataset.map(process_path_function, num_parallel_calls=self.autotune)

        return frame_dataset

    def convert_videos_to_frames(self):
        print("Converting videos... 0% done")
        counter = 0
        classes = os.listdir(self.videos_path)
        for classname in classes:
            files = os.listdir("{}/{}".format(self.videos_path, classname))
            file_paths = ["{}/{}/{}".format(self.videos_path, classname, filename) for filename in files]
            for path in file_paths:
                self._video_to_frames(path)
                counter += 1
                if counter % 60 == 0:
                    print("Converting videos... {}% done".format(int((counter/self.n_videos)*100)))
        print("Conversion done, frames stored in {}".format(self.frames_path))

    def _video_to_frames(self, video_path):
        video_path.replace('\\', '/')
        parts = video_path.split('/')
        classname = parts[-2]
        videoname = parts[-1].strip('.avi')

        output_dir = "{}/{}/{}".format(frames_path, classname, videoname)

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
            output_path = r"{}/{}/{}/{}.jpg".format(frames_path, classname, videoname, framename)
            image = cv2.resize(image, self.img_shape)
            output = cv2.imwrite(output_path, image)
            success, image = vidcap.read()
            count += 1
        return

    def generate_metadata(self):
        # TODO: Implement

        valid_frac = 0.15
        test_frac = 0.15
        metadata = {}
        for label_name in self.label_names:
            video_names = os.listdir(self.frame_path + '/' + label_name)
            n_videos = len(video_names)
            valid_start = -int(n_videos*(valid_frac + test_frac))
            test_start = -int(n_videos*(test_frac))
            train_videos = video_names[:valid_start]
            valid_videos = video_names[valid_start:test_start]
            test_videos = video_names[test_start:]
            metadata['train'] = {id: label_name for id in train_videos}
            metadata['valid'] = {id: label_name for id in valid_videos}
            metadata['test'] = {id: label_name for id in test_videos}
        return metadata


videos_path = './data/kth-actions/videos'
frames_path = './data/kth-actions/frame'
builder = DatasetBuilder(videos_path, frames_path, img_width=84, img_height=84, ms_per_frame=1000)
# metadata = builder.convert_videos_to_frames()

video_dataset = builder.make_video_dataset(metadata=None)
frame_dataset = builder.make_frame_dataset(metadata=None)
