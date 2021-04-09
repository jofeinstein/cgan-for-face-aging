import numpy
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization, Reshape, UpSampling2D, Conv2D, Activation


class Generator(tf.keras.Model):
    """ Generator Network """

    def __init__(self):
        super(Generator, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam()

        self.dense1 = Dense(2048)
        self.leakyrelu1 = LeakyReLU()
        self.dropout1 = Dropout(0.2)

        self.upsample1 = UpSampling2D(size=(2, 2))
        self.conv1 = Conv2D(filters=128, kernel_size=5, padding='same')
        self.batchnorm1 = BatchNormalization()
        self.leakyrelu2 = LeakyReLU()

        self.upsample2 = UpSampling2D(size=(2, 2))
        self.conv2 = Conv2D(filters=3, kernel_size=5, padding='same')
        self.tanh = Activation('tanh')

    def call(self, x):
        """ Passes input image through the network. """

        x = self.dense1(x)
        x = self.leakyrelu1(x)
        x = self.dropout1(x)

        x = self.reshape(x)
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.leakyrelu2(x)

        x = self.upsample1(x)
        x = self.conv2(x)
        x = self.tanh(x)

        return x

