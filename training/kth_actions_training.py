"""Basic script for training and evaluating a model on the kth actions dataset."""
import sys
sys.path.append('../')
sys.path.append('.')

import time

# Imports
from data_utils import kth_dataset_builder
from config import Config
import tensorflow as tf
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

plt.style.use('ggplot')

IMG_WIDTH = 120
IMG_HEIGHT = 120
N_CLASSES = 6

# Setup dataset builder
video_path = './data/kth-actions/video'
frame_path = './data/kth-actions/frame'
builder = kth_dataset_builder.DatasetBuilder(
    video_path, frame_path, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, ms_per_frame=100, max_frames=20)
builder.convert_videos_to_frames()
