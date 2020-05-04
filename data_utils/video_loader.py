"""Module for loading video files and tensors.

Example:
    loader = VideoLoader(video_folder_path='./data/something-something-mini-frame')
    videos = loader.load_all_videos()

"""
import sys
sys.path.append('../')
sys.path.append('.')

from config import Config

import tensorflow as tf
import os


class VideoLoader:
    """Used to load videos as numpy arrays."""

    def __init__(self, config):
        """Constructor."""
        self.video_folder_path = config.frame_path
        self.img_width = config.img_width
        self.img_height = config.img_height

    def load_all_videos(self, video_folder_path=None):
        """Load all videos in selected directory.

        Args:
            video_folder_path: Path of directory of videos encoded as image frames. Defaults to
            self.video_folder_path if omitted.

        Returns:
            Dict containing video samples. Key is video id, and maps to tensor of video data.
            Each tensor has shape (n, w, h, c) where n is the number of frames, w is the video
            width, h is the video height, and c is the number of channels. The value of n is
            variable. Label contains the integer label of the video.

        """
        if not video_folder_path:
            video_folder_path = self.video_folder_path

        video_names = self._get_all_video_names(video_folder_path)
        video_paths = ["{}/{}".format(video_folder_path, video_name) for video_name in video_names]

        videos = []
        for path in video_paths:
            videos.append(self._load_video(path))

        video_ids = [name.strip('.jpg') for name in video_names]
        samples = {}
        for id, video in zip(video_ids, videos):
            samples[int(id)] = video

        return samples

    def _get_all_video_names(self, video_folder_path):
        video_names = os.listdir(video_folder_path)
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
    config = Config()
    loader = VideoLoader(config)
    videos = loader.load_all_videos()
    vid = videos[2]
    print("--- Video Loader ---")
    print("Number of videos loaded: {}".format(len(videos)))
    print("Example video:\nID: {}, type: {}, shape: {}".format(
        2, type(vid), vid.shape))
