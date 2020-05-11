import sys
sys.path.append('../')
sys.path.append('.')

import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Dense, Conv3D, MaxPool3D, Flatten, Dropout, BatchNormalization, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from data_utils import kth_dataset_builder

IMG_WIDTH, IMG_HEIGHT = 84, 84

def Imagenet(input_shape=(160, 160, 3), name ='inception', weights=False, trainable=False, include_top=False):
    # Create the base model pre-trained on imagenet
    # the pretrained models come with input_shape = (160, 160, 3)
    # include_top=False removes the classification layer on top.
    # returned model returns a feature vector of dimension D from a given image
    if name=='mobilnet':
        # D = 1280
        base_model = tf.keras.applications.MobileNet(
                                                input_shape=input_shape,
                                                include_top=include_top,
                                                weights='imagenet')
    elif name=='inception':
        # D = 2048
        base_model = tf.keras.applications.InceptionV3(
                                                input_shape=input_shape,
                                                include_top=include_top,
                                                weights='imagenet')
    else: raise ValueError(name+" is not a valid model name. Give mobilnet or inception")

    base_model.trainable=trainable
    return base_model

def load_basic_cnn_lite():
    checkpoint_path = "./models/checkpoints/basic_cnn_lite"
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(84, 84, 1)),
        tf.keras.layers.Conv2D(32,
                               4,
                               padding='same', activation='relu',
                               input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(56,
                               5,
                               padding='same', activation='relu',
                               input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(72,
                               5,
                               padding='same', activation='relu',
                               input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(56,
                               4,
                               padding='same', activation='relu',
                               input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D()
        ])
    model.load_weights(checkpoint_path)
    model.trainable = False
    # model.summary()

    return model

def Video_Feature_Extractor(base_model):
    # returns an [BATCH x FRAME_NR x D] tensor where D is the dimension of the features
    feature_extractor = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D()
        ])

    if len(base_model.layers[0].input_shape) == 1:
        input_shape = base_model.layers[0].input_shape[0]
    else:
        input_shape = base_model.layers[0].input_shape


    features = TimeDistributed(
        layer = feature_extractor,
        input_shape = input_shape
        )

    return  tf.keras.Sequential([features])

def LSTM_Video_Classifier(features, class_nr, optimizer='adam'):
    # model
    full_model = tf.keras.Sequential([
        features,
        Dense(128, kernel_initializer="he_normal"),
        LSTM(512, input_shape=(None,128)),
        # Dense(512, kernel_initializer="he_normal"),
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

def AVG_Video_Classifier(features, class_nr, optimizer='adam'):
    # model
    full_model = tf.keras.Sequential([
        features,
        GlobalAveragePooling1D(),
        Dense(2048, kernel_initializer="he_normal"),
        Dense(class_nr, kernel_initializer="he_normal")])

    #compile model
    full_model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return full_model



# Doc:
"""
Imagenet(): This function allows you to load a base model pretrained on Imagenet. (either mobilnet or inception).

Video_Feature_Extractor(): allows you to extract features from video frames by returning:
                           [BATCH x FRAME_NR x D] tensor where D is the dimension of the features

LSTM_Video_Classifier(): returns an example model on how to use the returned frame features to train a RNN type network.

AVG_Video_Classifier(): returns an example model on how to use the returned frame features to train a simple MLP classifier.
"""

if __name__ == "__main__":
    # 1) Setup basic_cnn model
    # basic_cnn = load_basic_cnn_lite()
    # basic_cnn_extractor = Video_Feature_Extractor(basic_cnn)
    # basic_cnn_classifier = LSTM_Video_Classifier(features=basic_cnn_extractor, class_nr=6, optimizer=RMSprop(lr=0.0001))

    # 2) Setup mobilnet model
    mobilnet = Imagenet(name='mobilnet')
    mobilnet_extractor = Video_Feature_Extractor(mobilnet)
    mobilnet_classifier = LSTM_Video_Classifier(features=mobilnet_extractor, class_nr=6)

    # 3) Setup inception model
    inception = Imagenet(input_shape=(160, 160, 3), name='inception')
    inception_extractor = Video_Feature_Extractor(inception)
    inception_classifier = LSTM_Video_Classifier(features=inception_extractor, class_nr=6)

    # 4) Prep data
    video_path = './data/kth-actions/video'
    frame_path = './data/kth-actions/frame'

    builder = kth_dataset_builder.DatasetBuilder(
        video_path, frame_path, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, ms_per_frame=100, max_frames=20)
    metadata = builder.generate_metadata()

    train_ds = builder.make_video_dataset(metadata=metadata['train'])
    valid_ds = builder.make_video_dataset(metadata=metadata['valid'])
    test_ds = builder.make_video_dataset(metadata=metadata['test'])

    def format_example(image, label):
        image = tf.repeat(image, 3, axis=3)
        image = tf.image.resize(image, (160, 160))
        return image, label

    train_ds_scaled = train_ds.map(format_example)
    valid_ds_scaled = valid_ds.map(format_example)
    test_ds_scaled = test_ds.map(format_example)

    inception_classifier.fit(train_ds_scaled.shuffle(100).batch(25).prefetch(1), validation_data=valid_ds_scaled.batch(1), epochs=10)
