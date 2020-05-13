"""Script for tuning inception model on KTH actions dataset."""

# ======= IMPORTS ==========
import tensorflow as tf

from tensorflow import keras
import sklearn
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    tf.get_logger().setLevel('INFO')
    sys.path.insert(0, os.path.dirname('.'))
    sys.path.insert(0, os.path.dirname('../'))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

from data_utils import video_to_frames
from data_utils import metadata_loader
from data_utils.kth_dataset_builder import DatasetBuilder


def train_model():
    # ======= DATA ==========

    video_path = './data/kth-actions/video'
    frame_path = './data/kth-actions/frame'
    builder = DatasetBuilder(video_path, frame_path, img_width=120,
                             img_height=120, ms_per_frame=1000, max_frames=16)
    metadata = builder.generate_metadata()

    train_ds = builder.make_frame_dataset(metadata=metadata['train'])
    valid_ds = builder.make_frame_dataset(metadata=metadata['valid'])
    test_ds = builder.make_frame_dataset(metadata=metadata['test'])

    IMG_SIZE = 160  # All images will be resized to 160x160
    IMG_SHAPE = [IMG_SIZE, IMG_SIZE, 3]

    def format_example(image, label):
        image = tf.repeat(image, 3, axis=2)
        image = tf.image.resize(image, IMG_SHAPE[0:2])
        image.set_shape(IMG_SHAPE)
        return image, label

    train_ds_scaled = train_ds.map(format_example).batch(100).prefetch(1)
    valid_ds_scaled = valid_ds.map(format_example).batch(100)
    test_ds_scaled = valid_ds.map(format_example)


    # ====== BUILD MODEL ========

    base_model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(48, activation='relu', kernel_regularizer=keras.regularizers.l2(0.000144))(x)  #
    x = keras.layers.Dropout(0.0)(x)
    predictions = keras.layers.Dense(6, activation='softmax')(x)

    model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    #
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

    model.fit(train_ds_scaled.take(1000), epochs=1, validation_data=valid_ds_scaled.take(100))

    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

    model.fit(train_ds_scaled, epochs=4, validation_data=valid_ds_scaled)
    model.save("./models/trained_models/inception_tuned")
    model.save_weights("./models/checkpoints/inception_tuned")


def test_model():

    onehot_targets = [sample[1].numpy() for sample in valid_ds_scaled.take(30)]
    onehot_targets = np.concatenate(onehot_targets, axis=0)
    targets = np.argmax(onehot_targets, axis=1)

    model = tf.keras.models.load_model("./models/trained_models/inception_tuned")

    onehot_preds = model.predict(valid_ds_scaled.take(30))
    preds = np.argmax(onehot_preds, axis=1)
    conf = confusion_matrix(targets, preds)

    ax = plt.subplot()
    sns.heatmap(conf, annot=True, ax=ax, cmap=sns.cubehelix_palette(8), fmt='g',
        xticklabels=['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'],
        yticklabels=['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    ax.set_title('Confusion Matrix')
    plt.show()


def load_model(include_top=False):
    # model = tf.keras.models.load_model(".\\models\\trained_models\\inception_tuned")
    # model.save_weights("./models/checkpoints/inception_tuned/inception_tuned")
    base_model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)
    if include_top:
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(48, activation='relu', kernel_regularizer=keras.regularizers.l2(0.000144))(x)
        x = keras.layers.Dropout(0.0)(x)
        predictions = keras.layers.Dense(6, activation='softmax')(x)
        model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    else:
        model = keras.models.Model(inputs=base_model.input, outputs=base_model.output)
    model.load_weights("./models/checkpoints/inception_tuned/inception_tuned")
    model.trainable = False
    return model


if __name__ == "__main__":
    # train_model()
    # test_model()
    model = load_model()
    model.summary()


# ==== KERAS TUNING ====
#
# from kerastuner.tuners import Hyperband
#
# def build_model(hp):
#     base_model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)
#     x = base_model.output
#     x = keras.layers.GlobalAveragePooling2D()(x)
#     x = keras.layers.Dense(
#         hp.Int('hidden_nodes', 16, 512, step=32),
#         activation='relu',
#         kernel_regularizer=keras.regularizers.l2(
#             hp.Choice('reg', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5])))(x)
#     x = keras.layers.Dropout(hp.Float('drop', 0, 0.6, step=0.1))(x)
#     predictions = keras.layers.Dense(6, activation='softmax')(x)
#
#     model = keras.models.Model(inputs=base_model.input, outputs=predictions)
#     #
#     for layer in base_model.layers:
#         layer.trainable = False
#
#     model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
#
#     model.fit(train_ds_scaled.take(100), epochs=1, validation_data=valid_ds_scaled.take(100))
#
#     for layer in model.layers[:249]:
#        layer.trainable = False
#     for layer in model.layers[249:]:
#        layer.trainable = True
#
#     model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
#     return model
#
# tuner = Hyperband(
#     build_model,
#     objective='val_accuracy',
#     max_epochs=30,
#     hyperband_iterations=3)
#
# tuner.search(train_ds_scaled.take(100),
#              validation_data=valid_ds_scaled.take(50),
#              epochs=2,
#              callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])
#
# best_model = tuner.get_best_models(1)[0]
# best_hyperparams = tuner.get_best_hyperparameters(1)[0]
# print(best_hyperparams.values)
# best_model.save(".\\models\\trained_models\\inception_tuned")
