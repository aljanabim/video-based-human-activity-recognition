import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv3D, MaxPool3D, Flatten, Dropout,BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2



def C3D_model(input_shape, n_classes):
    # input_shape = (config.max_frames, config.img_width, config.img_height, 3)
    # nb_classes = config.n_classes


    weight_decay = 0.005
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
    x1 = Dense(n_classes, kernel_regularizer=l2(weight_decay))(x2)
    # x = Activation('softmax')(x)
    # out = concatenate([x1, x2, x3, x4], axis=-1)
    out = Activation('softmax')(x1)
    model = tf.keras.Model(inputs, out)

    model.compile(optimizer=config.optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=config.metrics)

    return model 

def small_c3d(input_shape, n_classes):
    #input_shape = (50, 84, 84,1)
    # n_classes = 6
    model = tf.keras.Sequential()

    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=input_shape, padding="same", activation='relu', name="1"))
    model.add(MaxPool3D(pool_size=(2, 2, 1), strides=(1,2,2), padding="valid", name="2"))
    model.add(Dropout(0.25, name="3"))


    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3), activation='relu', name="4"))
    model.add(MaxPool3D(pool_size=(3, 3, 3), padding="same", name="5"))
    model.add(Dropout(0.25, name="6"))

    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3), activation='relu', name="7"))
    model.add(MaxPool3D(pool_size=(3, 3, 3),padding="same", name="8"))
    model.add(Dropout(0.25, name="9"))


    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3), activation='relu', name="10"))
    model.add(MaxPool3D(pool_size=(3, 3, 3), padding="same", name="11"))
    model.add(Dropout(0.25, name="12"))



    model.add(Flatten(name="13"))
    model.add(Dense(64, activation='relu', name="14"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax', name="15"))
    model.summary()
    model.compile(optimizer='rmsprop',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    return model

def bigger_c3d(input_shape, n_classes):
    #input_shape=(85,200,120,1)
    
    inp = Input(shape=input_shape)


    x = Conv3D(64, kernel_size=(3,3,3), strides=(1, 1, 1), padding='valid')(inp)
    x = MaxPool3D(pool_size=(1, 2, 2), strides=(1,2,2), padding='valid')(x)

    x = Conv3D(64, kernel_size=(3,3,3), strides=(1, 1, 1), padding='valid')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2), padding='valid')(x)

    x = Conv3D(64, kernel_size=(3,3,3), strides=(1, 1, 1), padding='valid')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2), padding='valid')(x)

    x = Flatten()(x)
    x = Dense(256,activation='relu')(x)
    y = Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inp, y)
    model.summary()



    model.compile(optimizer='rmsprop',
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    return model


def main():
    import sys
    sys.path.append('../')
    sys.path.append('.')
    from config import Config
    config = Config()
    input_shape = (config.max_frames, config.img_width, config.img_height, 3)
    nb_classes = config.n_classes
    model = C3D_model(input_shape, nb_classes)
    print(model.summary())

if __name__ == "__main__":
    main()