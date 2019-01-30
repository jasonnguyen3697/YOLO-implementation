import keras
from keras import backend as k
from keras.layers import Conv2D, AvgPool2D, BatchNormalization, MaxPool2D, Dense, Activation, Input, LeakyReLU, Reshape
from keras.models import Model
import tensorflow as tf
import numpy

def preTrainX(samples, rows, cols):
    """This method creates the convolution neural network model of YOLO algorithm"""
    XInput = Input(shape=(samples, rows, cols, 3))

    X = Conv2D(filters=64, kernel_size=7, strides=2)(XInput)
    X = LeakyReLU(alpha=0.1)(X)
    X = MaxPool2D(pool_size=2, strides=2)(X)
    X = LeakyReLU(alpha=0.1)(X)

    X = Conv2D(filters=192, kernel_size=3, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = MaxPool2D(pool_size=2, strides=2)(X)
    X = LeakyReLU(alpha=0.1)(X)

    X = Conv2D(filters=128, kernel_size=1, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=256, kernel_size=3, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=256, kernel_size=1, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=512, kernel_size=3, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = MaxPool2D(pool_size=2, strides=2)(X)
    X = LeakyReLU(alpha=0.1)(X)

    X = Conv2D(filters=256, kernel_size=1, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=512, kernel_size=3, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=256, kernel_size=1, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=512, kernel_size=3, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=256, kernel_size=1, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=512, kernel_size=3, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=256, kernel_size=1, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=512, kernel_size=3, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=512, kernel_size=1, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=1024, kernel_size=3, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = MaxPool2D(pool_size=2, strides=2)(X)
    X = LeakyReLU(alpha=0.1)(X)

    X = Conv2D(filters=512, kernel_size=1, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=1024, kernel_size=3, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=512, kernel_size=1, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=1024, kernel_size=3, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=1024, kernel_size=3, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=1024, kernel_size=3, strides=2)(X)
    X = LeakyReLU(alpha=0.1)(X)

    #X = Conv2D(filters=1024, kernel_size=3, padding='same')(X)
    #X = LeakyReLU(alpha=0.1)(X)
    #X = Conv2D(filters=1024, kernel_size=3, padding='same')(X)
    #X = LeakyReLU(alpha=0.1)(X)

    #X = Dense(units=4096)(X)
    #X = LeakyReLU(alpha=0.1)(X)
    #X = Dense(units=1470)(X)
    #X = Reshape((7, 7, 30))(X)

    #model = Model(inputs=XInput, outputs=X)

    return X