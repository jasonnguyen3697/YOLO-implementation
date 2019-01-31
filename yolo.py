import keras
from keras import backend as k
from keras.regularizers import l2
from keras.layers import Conv2D, AvgPool2D, BatchNormalization, MaxPool2D, Dense, Activation, Input, LeakyReLU, Reshape
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy

def loadData(dim):
    """Load images according to the desired dimensions"""

    #create data generator
    datagen = ImageDataGenerator(validation_split=0.1, rescale=1./255)

    train_generator = datagen.flow_from_directory('Object Recognition Dataset/Training set/256_ObjectCategories', target_size=dim, batch_size=16, subset='training')
    val_generator = datagen.flow_from_directory('Object Recognition Dataset/Training set/256_ObjectCategories', target_size=dim, batch_size=16, subset='validation')

    return train_generator, val_generator

def preTrainModel(rows, cols):
    """This method creates the first 20 layers of the YOLO convolutional network for pre-training"""

    XInput = Input(shape=(rows, cols, 3))

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
    X = LeakyReLU(alpha=0.1, name='pre_train_output')(X)

    X = AvgPool2D(pool_size=2, strides=2)(X)
    X = Dense(units=4096, kernel_regularizer=l2())(X)

    model = Model(inputs=XInput, outputs=X)

    return model

def yoloModel(preTrainModel):
    """This method creates the rest of the YOLO convolutional neural network using pre-trained model"""

    X = Conv2D(filters=1024, kernel_size=3, padding='same')(preTrainModel.get_layer('pre_train_output').output)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(filters=1024, kernel_size=3, padding='same')(X)
    X = LeakyReLU(alpha=0.1)(X)

    X = Dense(units=4096)(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dense(units=1470)(X)
    X = Reshape((7, 7, 30))(X)

    model = Model(inputs=preTrainModel.input, outputs=X)

    return model

#def loss(YTrain, YPred):