# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel('INFO')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import os
import sys
sys.path.insert(0, os.path.dirname('.'))
sys.path.insert(0, os.path.dirname('../'))

import plot_utils
from data_utils import video_to_frames
from data_utils import metadata_loader
from data_utils.kth_dataset_builder import DatasetBuilder

from training import custom_cnn
from training import tuned_inception

from models.IMAGENET import Imagenet, Video_Feature_Extractor
from models.IMAGENET import AVG_Video_Classifier, LSTM_Video_Classifier

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, Conv3D, MaxPool3D, Flatten, Dropout, BatchNormalization, LSTM, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy

# Load Dataset
USE_TRIMMED = False  # use the trimmed larger data set of KTH videos

if USE_TRIMMED:
    video_path = '../data/kth-actions/video_trimmed'
    frame_path = '../data/kth-actions/frame_trimmed'
else:
    video_path = './data/kth-actions/video'
    frame_path = './data/kth-actions/frame'
# Setup builder
# video_path = '../data/kth-actions/video'
# frame_path = '../data/kth-actions/frame'
builder = DatasetBuilder(video_path, frame_path, img_width=120,
                         img_height=120, ms_per_frame=1000, max_frames=16)

# Convert videos and generate metadata
# builder.convert_videos_to_frames()
metadata = builder.generate_metadata()

# Build datasets
train_ds = builder.make_video_dataset(metadata=metadata['train'])
valid_ds = builder.make_video_dataset(metadata=metadata['valid'])

# Preprocess dataset
IMG_SIZE = 160  # All images will be resized to 160x160
IMG_SHAPE = [IMG_SIZE, IMG_SIZE, 3]


def format_example(image, label):
    image = tf.repeat(image, 3, axis=3)
    image = tf.image.resize(image, IMG_SHAPE[0:2])
    image.set_shape([None] + IMG_SHAPE)
    return image, label

train_ds_single_channel = train_ds
valid_ds_single_channel = valid_ds

train_ds = train_ds.map(format_example)  # with trimmed we have 1675
valid_ds = valid_ds.map(format_example)  # with trimmed we have 359

# Print
for x, lab in valid_ds.take(1):
    print(x.shape, lab.shape)
print(train_ds)

# %%


def My_Video_Classifier(features, class_nr, optimizer='adam'):
    # model
    full_model = tf.keras.Sequential([
        features,
        LSTM(1024, input_shape=(None, 18432)),
        Dense(512, kernel_initializer="he_normal", activation='relu'),
        # Dropout(rate=0.2),
        # Dense(512, activation='relu'),
        Dense(class_nr)
    ])

    # for layer in full_model.layers[:249]:
    #    layer.trainable = False
    # for layer in full_model.layers[249:]:
    #    layer.trainable = True

    # compile model
    full_model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )




    return full_model


# Base model (returns pretrained frozen base model trained on Imagenet)
# custom_cnn = custom_cnn.load_model()
tuned_inception = tuned_inception.load_model(include_top=False)
tuned_inception.summary()
# inception = Imagenet(input_shape=IMG_SHAPE, name='inception')

# Feature Extractor (Has output (NR_FRAME x D) where D is feature dimension)
feature_ex = Video_Feature_Extractor(tuned_inception)
# feature_ex.summary()

# LSTM Clasifier
model = My_Video_Classifier(features=feature_ex, class_nr=6)
# model.summary()
# %%
history = model.fit(train_ds.shuffle(210).batch(14).prefetch(1),
          validation_data=valid_ds.batch(1), epochs=50)

plot_utils.kth_confusion_matrix(model, valid_ds.batch(1))





# SO far 70 epoch

# # %%
# model.save('trained_models/LSTM_50epochs_trimmed')
# # %%
# avg = 0
# for i in range(100):
#     avg += model.evaluate(valid_ds.batch(1))[1]
# print(avg / 100)

# Last avg with 50 epochs and untrimmed data set gives 0.6695555579662323
# Last avg with 50 epochs and trimmed data set gives 0.7825069636106491
