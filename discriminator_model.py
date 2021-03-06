import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, LeakyReLU, BatchNormalization, Embedding, Reshape


class Discriminator(tf.keras.Model):
    def __init__(self):

        super(Discriminator, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam()

        self.embedding = Embedding(input_dim=6, output_dim=256)
        self.dense_e = Dense(32 * 32)
        self.reshape = Reshape((32, 32, 6))

        self.conv1 = Conv2D(64, kernel_size=3, strides=2, padding='same')
        self.leakyrelu1 = LeakyReLU()

        self.conv2 = Conv2D(128, kernel_size=3, strides=2, padding='same')
        self.batchnorm1 = BatchNormalization()
        self.leakyrelu2 = LeakyReLU()

        self.conv3 = Conv2D(256, kernel_size=3, strides=2, padding='same')
        self.batchnorm2 = BatchNormalization()
        self.leakyrelu3 = LeakyReLU()

        self.conv4 = Conv2D(512, kernel_size=3, strides=2, padding='same')
        self.batchnorm3 = BatchNormalization()
        self.leakyrelu4 = LeakyReLU()

        self.flatten = Flatten()
        self.dense = Dense(1, activation='sigmoid')


    def call(self, input):
        """ Passes input image through the network. """

        image, label = input

        x = self.conv1(image)
        x = self.leakyrelu1(x)

        label = self.embedding(label)
        label = self.dense_e(label)
        label = self.reshape(label)

        # check concatenation
        x = tf.concat([x, label], axis=3)

        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.leakyrelu2(x)

        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.leakyrelu3(x)

        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.leakyrelu4(x)

        x = self.flatten(x)
        x = self.dense(x)

        return x

    def loss_function(self, y_true, y_pred):
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        return loss_fn(y_true, y_pred)


