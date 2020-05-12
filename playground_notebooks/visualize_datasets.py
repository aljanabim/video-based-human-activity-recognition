from data_utils.kth_dataset_builder import DatasetBuilder
import cv2 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Setup builder
video_path = './data/kth-actions/video'
frame_path = './data/kth-actions/frame'
builder = DatasetBuilder(video_path, frame_path, img_width=120, img_height=120, ms_per_frame=1000)

# Convert videos and generate metadata
#builder.convert_videos_to_frames()
metadata = builder.generate_metadata()

# Build datasets
video_dataset = builder.make_video_dataset(metadata=metadata['train'])
valid_dataset = builder.make_video_dataset(metadata=metadata['valid'])


# Verify that the datasets work
for vid, label in valid_dataset.take(20):
    print(vid.shape, label)


    builder.label_names[np.argmax(label.numpy())]



fig = plt.figure(figsize = (15,16))
fig.suptitle("Class"+builder.label_names[np.argmax(label.numpy())], fontsize=25)
for i,frame in enumerate(vid):
    plt.subplot(5,5,i+1)
    # plt.imshow(frame.numpy())
    plt.title(str(i+1))
    plt.imshow(tf.squeeze(frame).numpy())