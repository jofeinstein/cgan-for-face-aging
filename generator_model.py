import numpy
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization, UpSampling2D, Conv2D, Activation, Embedding, Reshape


class Generator(tf.keras.Model):
    """ Generator Network """

    def __init__(self):
        super(Generator, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam()

        self.reshape = Reshape((8,8,32))

        self.dense1 = Dense(2048)
        self.leakyrelu1 = LeakyReLU()
        self.dropout1 = Dropout(0.2)

        self.upsample1 = UpSampling2D(size=(2, 2))
        self.conv1 = Conv2D(filters=128, kernel_size=5, padding='same')
        self.batchnorm1 = BatchNormalization()
        self.leakyrelu2 = LeakyReLU()

        self.upsample2 = UpSampling2D(size=(2, 2))
        self.conv2 = Conv2D(filters=64, kernel_size=5, padding='same')
        self.batchnorm2 = BatchNormalization()
        self.leakyrelu3 = LeakyReLU()

        self.upsample3 = UpSampling2D(size=(2, 2))
        self.conv3 = Conv2D(filters=32, kernel_size=5, padding='same')
        self.batchnorm3 = BatchNormalization()
        self.leakyrelu4 = LeakyReLU()

        self.upsample4 = UpSampling2D(size=(2, 2))
        self.conv4 = Conv2D(filters=3, kernel_size=5, padding='same')
        self.tanh = Activation('tanh')

    def call(self, input):
        """ Passes input image through the network. """

        latent_vector, label = input

        x = tf.concat([latent_vector, label], axis=1)

        x = self.dense1(x)
        x = self.leakyrelu1(x)
        x = self.dropout1(x)

        x = self.reshape(x)

        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.leakyrelu2(x)

        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.leakyrelu3(x)

        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.leakyrelu4(x)

        x = self.upsample4(x)
        x = self.conv4(x)
        x = self.tanh(x)

        return x

    def loss_function(self, y_true, y_pred):
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        return loss_fn(y_true, y_pred)