import sys
sys.path.append('../')
sys.path.append('.')


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import os
import sys
sys.path.insert(0, os.path.dirname('../'))

from data_utils import video_to_frames
from data_utils import metadata_loader
from data_utils.kth_dataset_builder import DatasetBuilder

from models.IMAGENET import Imagenet, Video_Feature_Extractor
from models.IMAGENET import AVG_Video_Classifier, LSTM_Video_Classifier

# Setup builder
video_path = './data/kth-actions/video'
frame_path = './data/kth-actions/frame'
builder = DatasetBuilder(video_path, frame_path, img_width=84, img_height=84, ms_per_frame=1000, max_frames=16)

# Convert videos and generate metadata
#builder.convert_videos_to_frames()
metadata = builder.generate_metadata()

# Build datasets
train_ds = builder.make_video_dataset(metadata=metadata['train'])
valid_ds = builder.make_video_dataset(metadata=metadata['valid'])

# Preprocess dataset
IMG_SIZE = 160 # All images will be resized to 160x160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def format_example(image, label):
    image = tf.repeat(image,3,axis=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train_ds = train_ds.map(format_example)
valid_ds = valid_ds.map(format_example)

# Print
for x, lab in valid_ds.take(1):
    print(x.shape, lab.shape)
train_ds

# Training set
a = np.zeros(6)
for _, label in train_ds.as_numpy_iterator():
    a=a+label
print("Training set, cases for each class:",a)

# Valid Set
a = np.zeros(6)
for _, label in valid_ds.as_numpy_iterator():
    a=a+label
print("Validation set, cases for each class:",a)

from tensorflow.keras.layers import Input, Activation, Dense, Conv3D, MaxPool3D, Flatten, Dropout, BatchNormalization, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy

def My_Video_Classifier(features, class_nr, optimizer='adam'):
    # model
    full_model = tf.keras.Sequential([
        features,
        # Dense(128, kernel_initializer="he_normal"),
        LSTM(512, input_shape=(None,128)),
        #Dense(512, kernel_initializer="he_normal"),
        Dropout(rate=0.4),
        Dense(class_nr)
        ])

    #compile model
    full_model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )
    return full_model

    # Base model (returns pretrained frozen base model trained on Imagenet)
inception = Imagenet(input_shape=IMG_SHAPE, name='inception')

# Feature Extractor (Has output (NR_FRAME x D) where D is feature dimension)
featuer_ex = Video_Feature_Extractor(inception)

# LSTM Clasifier
model = My_Video_Classifier(features=featuer_ex, class_nr=6)
model.summary()

model.fit(train_ds.shuffle(100).batch(25).prefetch(1), validation_data=valid_ds.batch(1), epochs=10)
