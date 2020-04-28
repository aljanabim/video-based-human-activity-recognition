"""Module for loading video files and tensors.

Example:
    loader = VideoLoader(dataset_root='./data/something-something-mini')
    videos = loader.load_all_videos()

"""

import tensorflow as tf
import os


class VideoLoader:
    """Used to load videos as numpy arrays."""

    def __init__(self, img_width=455, img_height=256,
                 dataset_root='./data/something-something-mini'):
        """Constructor."""
        self.dataset_root = dataset_root
        self.img_width = img_width
        self.img_height = img_height

    def load_all_videos(self, dir_path=None):
        """Load all videos in selected directory.

        Args:
            dir_path: Path of directory of videos encoded as image frames. Defaults to
            self.dataset_root if omitted.

        Returns:
            List of dicts containing video samples. Each dict has keys id, data, and label. Id
            contains integer id of the video. Data contains a tensor of video data. Each tensor has
            shape (n, w, h, c) where n is the number of frames, w is the video width, h is the video
            height, and c is the number of channels. The value of n is variable. Label contains the
            integer label of the video.

        """
        if not dir_path:
            dir_path = '{}-frame'.format(self.dataset_root)

        video_names = self._get_all_video_names(dir_path)
        video_paths = ["{}/{}".format(dir_path, video_name) for video_name in video_names]

        videos = []
        for path in video_paths:
            videos.append(self._load_video(path))

        video_ids = [name.strip('.jpg') for name in video_names]
        samples = []
        for id, video in zip(video_ids, videos):
            samples.append({'id': id, 'data': video})

        return samples

    def _get_all_video_names(self, dir_path):
        video_names = os.listdir(dir_path)
        return video_names

    def _load_video(self, frame_dir_path):
        frame_names = os.listdir(frame_dir_path)
        frame_paths = ["{}/{}".format(frame_dir_path, frame_name) for frame_name in frame_names]

        frames = []
        for path in frame_paths:
            frames.append(self._load_frame(path))
        video = tf.stack(frames)
        return video

    def _load_frame(self, frame_path):
        frame = tf.io.read_file(frame_path)
        frame = tf.image.decode_jpeg(frame, channels=3)
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        return tf.image.resize(frame, [self.img_width, self.img_height])


if __name__ == "__main__":
    loader = VideoLoader()
    videos = loader.load_all_videos()
    vid = videos[0]
    print("--- Video Loader ---")
    print("Number of videos loaded: {}".format(len(videos)))
    print("Example video:\nID: {}, type: {}, shape: {}".format(
        vid['id'], type(vid['data']), vid['data'].shape))
