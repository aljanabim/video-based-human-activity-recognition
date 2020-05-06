import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv3D, MaxPool3D, Flatten, Dropout,BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
sys.path.append('.')
from config import Config

def C3D_model(config):
    input_shape = (config.max_frames, config.img_width, config.img_height, 3)
    weight_decay = 0.005
    nb_classes = config.n_classes

    inputs = Input(input_shape)
    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x4 = Flatten()(x)
    x3 = Dense(2048, activation='relu', kernel_regularizer=l2(weight_decay))(x4)
    # x = Dropout(0.5)(x)
    x2 = Dense(2048, activation='relu', kernel_regularizer=l2(weight_decay))(x3)
    # x = Dropout(0.5)(x)
    x1 = Dense(nb_classes, kernel_regularizer=l2(weight_decay))(x2)
    # x = Activation('softmax')(x)
    # out = concatenate([x1, x2, x3, x4], axis=-1)
    out = Activation('softmax')(x1)
    model = tf.keras.Model(inputs, out)

    model.compile(optimizer=config.optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=config.metrics)

    return model 


def main():
    config = Config()
    model = c3d_model(config)
    print(model.summary())

if __name__ == "__main__":
    main()