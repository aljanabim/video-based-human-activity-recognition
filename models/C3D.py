import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv3D, MaxPool3D, Flatten, Dropout,BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2

def C3D_model(input_shape, n_classes):
    # Returns the C3D model as described in the paper (up to the compiling parameters)
    weight_decay = 0.005
    inputs = Input(input_shape)
    x = Conv3D(
        filters=64,
        kernel_size=(3, 3, 3),
        strides=(1, 1, 1),
        padding='same',
        activation='relu',
        kernel_regularizer=l2(weight_decay))(inputs)

    x = MaxPool3D(
        pool_size=(2, 2, 1),
        strides=(2, 2, 1),
        padding='same')(x)

    filters = [128,128,256,256]
    for filter in filters:
        x = Conv3D(
            filters=filter,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            activation='relu',
            kernel_regularizer=l2(weight_decay))(x)

        x = MaxPool3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            padding='same')(x)

    x4 = Flatten()(x)

    x3 = Dense(
        units=2048,
        activation='relu',
        kernel_regularizer=l2(weight_decay))(x4)
    # x3 = Dropout(0.5)(x4)
    x2 = Dense(
        units=2048,
        activation='relu',
        kernel_regularizer=l2(weight_decay))(x3)
    # x2 = Dropout(0.5)(x3)
    x1 = Dense(
        units=n_classes,
        kernel_regularizer=l2(weight_decay))(x2)
    # out = concatenate([x1, x2, x3, x4], axis=-1)  # needed to extracts features from the last 4 layers
    model = tf.keras.Model(inputs, x1)

    model.compile(
        optimizer = 'rmsprop',
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy'])
    return model 

def Small_C3D(input_shape, n_classes):
    # try out models( much smaller than the original one)
    model = tf.keras.Sequential()

    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=input_shape, padding="same", activation='relu', name="1"))
    model.add(MaxPool3D(pool_size=(2, 2, 1), strides=(1,2,2), padding="valid", name="2"))
    model.add(Dropout(0.25, name="3"))

    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3), activation='relu', name="4"))
    model.add(MaxPool3D(pool_size=(3, 3, 3), padding="same", name="5"))
    model.add(Dropout(0.25, name="6"))

    model.add(Conv3D(128, padding="same", kernel_size=(3, 3, 3), activation='relu', name="7"))
    model.add(MaxPool3D(pool_size=(3, 3, 3),padding="same", name="8"))
    model.add(Dropout(0.25, name="9"))

    model.add(Conv3D(256, padding="same", kernel_size=(3, 3, 3), activation='relu', name="10"))
    model.add(MaxPool3D(pool_size=(3, 3, 3), padding="same", name="11"))
    model.add(Dropout(0.25, name="12"))

    model.add(Flatten(name="13"))

    model.add(Dense(1024, activation='relu', name="14"))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation='softmax', name="15"))

    model.compile(
        optimizer='rmsprop',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])
    return model

def Bigger_C3D(input_shape, n_classes):
    # try out models( much smaller than the original one)
    inp = Input(shape=input_shape)
    x = Conv3D(64, kernel_size=(3,3,3), strides=(1, 1, 1), padding='valid')(inp)
    x = MaxPool3D(pool_size=(1, 2, 2), strides=(1,2,2), padding='valid')(x)

    x = Conv3D(128, kernel_size=(3,3,3), strides=(1, 1, 1), padding='valid')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2), padding='valid')(x)

    x = Conv3D(256, kernel_size=(3,3,3), strides=(1, 1, 1), padding='valid')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2), padding='valid')(x)

    x = Flatten()(x)
    y = Dense(2048,activation='relu')(x)

    model = tf.keras.Model(inp, y)
    model.summary()

    model.compile(optimizer='rmsprop',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model


# Doc
"""
Functions that return 3 different 3D convolution models. 
Be careful they are very computaional intensiv.
For example, C3D_model() presented in [1] has 90M parameters.

[1] Learning Spatiotemporal Features with 3D Convolutional Networks
"""
# if __name__ == "__main__":

#     input_shape = (50, 84, 84, 1) # (max_frames, width, height, channels)
#     n_classes = 6

#     #Get the c3d model as shown in the paper
#     # model = C3D_model(input_shape=input_shape, n_classes=n_classes)
#     # print(model.summary())

#     #Get a smaller c3d model
#     model = Bigger_C3D(input_shape=input_shape, n_classes=n_classes)
#     print(model.summary()) 
    
#     #Get an even smaller c3d model
#     model = Small_C3D(input_shape=input_shape, n_classes=n_classes)
#     print(model.summary())