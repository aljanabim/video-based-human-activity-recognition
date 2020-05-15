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
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

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
    print("Using untrimmed")
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
    if use_SVM:
        dense = tf.keras.Sequential([keras.layers.GlobalAveragePooling1D(),
                                     keras.layers.Dense(N_CLASSES,
                                                        kernel_regularizer=keras.regularizers.l2(0.002))])
        loss = tf.losses.Hinge()
        optimizer = 'RMSprop'
    else:
        dense = tf.keras.Sequential([keras.layers.LSTM(1024),
                                     keras.layers.Dense(N_CLASSES)])
        loss = tf.losses.CategoricalCrossentropy(from_logits=True)
        optimizer = 'adam'

    feature_extractor = Video_Feature_Extractor(base_model)

    full_model = tf.keras.Sequential([
        feature_extractor,
        dense
    ])

    full_model.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=['accuracy'])
    return full_model


full_model = get_inception_model(load_tuned=False, use_SVM=True)
full_model_tuned = get_inception_model(load_tuned=True, use_SVM=True)

# %% history storage
history_saved = []
history_tuned_saved = []

# %% TRAIN THE MODELS
history = full_model.fit(train_ds.shuffle(80).batch(14).prefetch(1),
                         validation_data=valid_ds.batch(1), epochs=70)
history_tuned = full_model_tuned.fit(train_ds.shuffle(80).batch(14).prefetch(1),
                                     validation_data=valid_ds.batch(1), epochs=70)
history_saved.append(history)
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

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 3.5))
    axs[0].plot(acc, label="Train Acc")
    axs[0].plot(acc_val, label="Valid Acc")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].set(xlim=(0, len(acc)), ylim=(0, 1))
    axs[0].set_title("Model accuracy")
    axs[0].legend()

    axs[1].plot(loss, label="Train Loss")
    axs[1].plot(loss_val, label="Valid Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].set(xlim=(0, len(acc)))
    axs[1].set_title("Model Loss")
    axs[1].legend(loc="bottom left")
    fig.tight_layout(pad=2.0)
    fig.savefig('plots/SVM_70epochs_untrimmed_tuned_acc_loss.pdf')
    plt.show()
    cmat = confusion_matrix(y_test, y_pred)
    cmat_plot = plot_confusion_matrix(conf_mat=cmat, figsize=(5, 5),
                                      class_names=CLASS_NAMES)
    plt.savefig('plots/SVM_70epochs_untrimmed_tuned_cmat.pdf',
                bbox_inches='tight')
    return [acc, acc_val, loss, loss_val]


# %% GET TEST LABLES
y_test = [np.argmax(l.numpy()) for _, l in test_ds.batch(1)]
# %% GET NETWORK PREDICTIONS
y_pred = full_model.predict_classes(test_ds.batch(1))
y_pred_tuned = full_model_tuned.predict_classes(test_ds.batch(1))
# %% GET PLOT
logs = plot(history_tuned_saved, y_pred_tuned, y_test)


# %% SAVE LOGS
save_logs_to = './logs/SVM_70epochs_untrimmed_tuned.pkl'
overwrite = False
save = True
if os.path.exists(save_logs_to):
    save = False
    while True:
        print("File already exits, please try another path or filename")
        user_pref = input("Would you like to overwrite [y/N]")
        if user_pref == "n" or user_pref == "N":
            overwrite = False
            break
        if user_pref == "y" or user_pref == "Y":
            overwrite = True
            break
        else:
            print('Invalid input, please use "y" for yes and "N" for no')

if overwrite or save:
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
