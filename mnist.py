

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense,Reshape,Conv2DTranspose,BatchNormalization,
                                     LeakyReLU,Conv2D,Dropout,Flatten,Input)
from tensorflow.keras.datasets import mnist

#import keras 
#from keras.models import Sequential
#from keras.layers import (Dense,Reshape,Conv2DTranspose,BatchNormalization,
#                                     LeakyReLU,Conv2D,Dropout,Flatten,Input)
#from keras.datasets import mnist




#encoder/decoder shapes are as seen here : https://github.com/keras-team/keras/blob/master/examples/mnist_acgan.py
def build_encoder(input_shape,output_size=256):
    input_layer = keras.layers.Input(shape=input_shape,name='encoder_input')
    x = Conv2D(32, 3, padding='same', strides=2,
                   input_shape=(28, 28, 1),)(input_layer)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, 3, padding='same', strides=1)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, 3, padding='same', strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(output_size, 3, padding='same', strides=1)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    
    
    return keras.models.Model([input_layer],[x],name='encoder')
    
def build_decoder(input_shape):
    input_layer = keras.layers.Input(shape=input_shape,name='encoder_input')
    x = Dense(384, activation='relu',)(input_layer)#TODO: activation
    x = Dense(3 * 3 * 384, activation='relu')(x)
    x = Reshape((3, 3, 384))(x)

    # upsample to (7, 7, ...)
    x = Conv2DTranspose(192, 5, strides=1, padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)

    # upsample to (14, 14, ...)
    x = Conv2DTranspose(96, 5, strides=2, padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)

    # upsample to (28, 28, ...)
    x = Conv2DTranspose(1, 5, strides=2, padding='same',
                            activation='tanh',
                            kernel_initializer='glorot_normal',
                            name='decoder_output')(x)
    return keras.models.Model([input_layer],[x],name='decoder')
    
def get_data():
    if keras.backend.image_data_format() == 'channels_last':
        ch_axis = 3
    else:
        ch_axis = 1
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32)-127.5)  / 127.5
    x_train = np.expand_dims(x_train, axis=ch_axis)

    x_test = (x_test.astype(np.float32)-127.5)/ 127.5
    x_test = np.expand_dims(x_test, axis=ch_axis)
    
    return x_train,x_test,y_train,y_test