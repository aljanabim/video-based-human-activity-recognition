import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Dense, Conv3D, MaxPool3D, Flatten, Dropout, BatchNormalization, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2


def Imagenet(input_shape=(160, 160, 3), name='inception', weights=False, trainable=False, include_top=False):
    # Create the base model pre-trained on imagenet
    # the pretrained models come with input_shape = (160, 160, 3)
    # include_top=False removes the classification layer on top.
    # returned model returns a feature vector of dimension D from a given image
    if name == 'mobilnet':
        # D = 1280
        base_model = tf.keras.applications.MobileNet(
            input_shape=input_shape,
            include_top=include_top,
            weights='imagenet')
    elif name == 'inception':
        # D = 2048
        base_model = tf.keras.applications.InceptionV3(
            input_shape=input_shape,
            include_top=include_top,
            weights='imagenet')
    elif name == 'vgg':
        # D = 2048
        base_model = tf.keras.applications.VGG16(
            input_shape=input_shape,
            include_top=include_top,
            weights='imagenet')
    elif name == 'resnet':
        # D = 2048
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=include_top,
            weights='imagenet')
    elif name == 'resnet_v2':
        # D = 2048
        base_model = tf.keras.applications.ResNet50V2(
            input_shape=input_shape,
            include_top=include_top,
            weights='imagenet')
    else:
        raise ValueError(
            name + " is not a valid model name. Give mobilnet or inception")

    base_model.trainable = trainable
    return base_model


def Video_Feature_Extractor(base_model):
    # returns an [BATCH x FRAME_NR x D] tensor where D is the dimension of the features
    feature_extractor = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])

    features = TimeDistributed(
        layer=feature_extractor,
        input_shape=base_model.layers[0].input_shape[0]
    )

    return tf.keras.Sequential([features])


def LSTM_Video_Classifier(features, class_nr, optimizer='adam'):
    # model
    full_model = tf.keras.Sequential([
        features,
        LSTM(128, input_shape=(None, 256)),
        Dense(512, kernel_initializer="he_normal"),
        BatchNormalization(),
        Dense(class_nr)
    ])

    # compile model
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

    # compile model
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
    # 1)Get pretrained Base Model (either inception or mobilnet which is untrainable per default)
    # inception = Imagenet(name='inception')
    # mobilnet = Imagenet(name='mobilnet')
    resnet = Imagenet(name='vgg')
    # inception.summary()

    # # 2)Get the Featuer Extractor (calculates one feature for each frame using the base model)
    # featuer_ex1 = Video_Feature_Extractor(inception)
    # featuer_ex2 = Video_Feature_Extractor(mobilnet)
    featuer_ex3 = Video_Feature_Extractor(resnet)
    # featuer_ex.summary()

    # 3)Get the Video Classifiers (add a classifying model on top which can trained on given data)
    lstm_video_classifier = LSTM_Video_Classifier(
        features=featuer_ex3, class_nr=6, optimizer=RMSprop(lr=0.0001))
    lstm_video_classifier.summary()

    # avg_video_classifier = AVG_Video_Classifier(features=featuer_ex2, class_nr=6, optimizer=RMSprop(lr=0.0001))
    # avg_video_classifier.summary()

    # ####
    # # How to save
    # print("PREDICT:",lstm_video_classifier.predict(np.ones((1,16,160,160,3))))
    # print("SAVING")
    # # Save only weights
    # lstm_video_classifier.save_weights("bullshit/")
    # lstm_video_classifier = None

    # print("LOADING")
    # # Create model
    # inception = Imagenet(name='inception')
    # featuer_ex1 = Video_Feature_Extractor(inception)
    # model = LSTM_Video_Classifier(features=featuer_ex1, class_nr=6, optimizer=RMSprop(lr=0.0001))
    # # load weights
    # model.load_weights("bullshit/")
    # print("PREDICT:",model(np.ones((1,16,160,160,3))))
