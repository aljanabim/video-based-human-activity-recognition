import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import sys
sys.path.append('.')

from config import Config
from data_utils import video_to_frames
from data_utils import metadata_loader
from data_utils import dataset_builder
from data_utils.kth_dataset_builder import DatasetBuilder



# config = Config(root_path='./data/1000-videos', img_width=84, img_height=84, use_subfolders=True)
config = Config()

    # Setup builder
video_path = './data/kth-actions/video'
frame_path = './data/kth-actions/frame'
builder = DatasetBuilder(video_path, frame_path, img_width=84, img_height=84,max_frames=16)

# Convert videos and generate metadata
#builder.convert_videos_to_frames()
metadata = builder.generate_metadata()

# Build datasets
train_dataset = builder.make_video_dataset(metadata=metadata['train'])
valid_dataset = builder.make_video_dataset(metadata=metadata['valid'])

for x, lab in train_dataset.take(10):
    print(x.shape, lab.shape)
