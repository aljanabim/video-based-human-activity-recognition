'''
In this file we import
1. Untuned INCEPTION V3
2. Tuned INCEPTION V3

We then add and LSTM to both of these and train for 50 epochs with trimmed and untrimmed data

Lastly we generate plots over the
1. validation accuracy+loss
2. training accuracy+loss
3. confusion matrix
'''

# %% IMPORTS
import os
import sys
sys.path.insert(0, os.path.dirname('.'))
sys.path.insert(0, os.path.dirname('../'))

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import matplotlib.gridspec as gridspec
# pip install mlxtend
# from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, plot_confusion_matrix()

import tensorflow as tf
from tensorflow import keras
import pickle

from data_utils.kth_dataset_builder import DatasetBuilder
from models.IMAGENET import Imagenet, Video_Feature_Extractor

# %% LOAD DATA
USE_TRIMMED = True  # use the trimmed larger data set of KTH videos
IMG_SIZE = 160  # All images will be resized to 160x160
IMG_SHAPE = [IMG_SIZE, IMG_SIZE, 3]
MAX_FRAMES = 16
CLASS_NAMES = ['boxing', 'handclapping',
               'handwaving', 'jogging', 'running', 'walking']
N_CLASSES = len(CLASS_NAMES)
if USE_TRIMMED:
    video_path = '../data/kth-actions/video_trimmed'
    frame_path = '../data/kth-actions/frame_trimmed'
else:
    video_path = '../data/kth-actions/video'
    frame_path = '../data/kth-actions/frame'

builder = DatasetBuilder(video_path, frame_path, img_width=120,
                         img_height=120, ms_per_frame=100, max_frames=MAX_FRAMES)
metadata = builder.generate_metadata()

train_ds = builder.make_video_dataset(metadata['train'])
valid_ds = builder.make_video_dataset(metadata['valid'])
test_ds = builder.make_video_dataset(metadata['test'])

# %% DEF UTILITY FUNCTIONS


def format_example(image, label):
    image = tf.repeat(image, 3, axis=3)
    image = tf.image.resize(image, IMG_SHAPE[0:2])
    image.set_shape([None] + IMG_SHAPE)
    return image, label


train_ds = train_ds.map(format_example)
valid_ds = valid_ds.map(format_example)
test_ds = test_ds.map(format_example)

# %% LOADING THE TUNED AND UNTUNED INCEPTION V3 MODELS


def get_inception_model(load_tuned=False, use_SVM=False):
    base_model = Imagenet(input_shape=IMG_SHAPE, name='inception')
    if load_tuned:
        base_model.load_weights(
            "../models/checkpoints/inception_tuned/inception_tuned")
    feature_extractor = Video_Feature_Extractor(base_model)

    full_model = tf.keras.Sequential([
        feature_extractor,
        keras.layers.LSTM(1024),
        keras.layers.Dense(N_CLASSES)
    ])

    if use_SVM:
        loss = tf.losses.hinge_loss()
    else:
        loss = tf.losses.CategoricalCrossentropy(from_logits=True)

    full_model.compile(optimizer='adam',
                       loss=loss,
                       metrics=['accuracy'])
    return full_model


full_model = get_inception_model(load_tuned=False)
full_model_tuned = get_inception_model(load_tuned=True)

# %% history storage
history_saved = []
history_tuned_saved = []

# %% TRAIN THE MODELS
# history = full_model.fit(train_ds.shuffle(80).batch(14).prefetch(1),
#                          validation_data=valid_ds.batch(1), epochs=50)
history_tuned = full_model_tuned.fit(train_ds.shuffle(80).batch(14).prefetch(1),
                                     validation_data=valid_ds.batch(1), epochs=50)
# when done tot will be 120 epochs
# history_saved.append(history)
history_tuned_saved.append(history_tuned)
# %% PLOTTING


def plot(history, y_pred, y_test):
    acc = []
    acc_val = []
    loss = []
    loss_val = []
    for i in range(len(history)):
        acc += history[i].history['accuracy']
        acc_val += history[i].history['val_accuracy']
        loss += history[i].history['loss']
        loss_val += history[i].history['val_loss']
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Acc")
    plt.plot(acc_val, label="Valid Acc")
    plt.xlabel("Epoch")
    plt.xlim(0, len(acc))
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Model accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(loss_val, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.xlim(0, len(acc))
    plt.ylabel("Loss")
    plt.legend(loc="bottom left")
    plt.title("Model accuracy")
    plt.savefig('plots/LSTM_70epochs_tuned_acc_loss.pdf', bbox_inches='tight')
    plt.show()
    cmat = confusion_matrix(y_test, y_pred)
    cmat_plot = plot_confusion_matrix(conf_mat=cmat, figsize=(5, 5),
                                      class_names=CLASS_NAMES)
    plt.savefig('plots/LSTM_70epochs_cmat.pdf', bbox_inches='tight')
    return [acc, acc_val, loss, loss_val]


# %%
y_test = [np.argmax(l.numpy()) for _, l in test_ds.batch(1)]
y_pred_tuned = full_model_tuned.predict_classes(test_ds.batch(1))
# %%
logs = plot(history_tuned_saved, y_pred_tuned, y_test)

# %%
save_logs_to = './logs/LSTM_70epochs_tuned.pkl'
if os.path.exists(save_logs_to):
    print("File already exits, please try another path or filename")

else:
    print("Dumping the logs to ", save_logs_to)
    with open(save_logs_to, 'wb') as f:
        obj = {
            'acc': logs[0],
            'acc_val': logs[1],
            'loss': logs[2],
            'loss_val': logs[3],
            'y_test': y_test,
            'y_pred': y_pred_tuned}
        pickle.dump(obj, f)
