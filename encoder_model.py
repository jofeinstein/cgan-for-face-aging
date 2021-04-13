import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, BatchNormalization, Conv2D


class Encoder(tf.keras.Model):
    """ Encoder Network """

    def __init__(self):
        super(Encoder, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam()

        self.conv1 = Conv2D(filters=16, kernel_size=5, padding='same')
        self.leakyrelu1 = LeakyReLU()

        self.conv2 = Conv2D(filters=32, kernel_size=5, padding='same')
        self.batchnorm2 = BatchNormalization()
        self.leakyrelu2 = LeakyReLU()

        self.conv3 = Conv2D(filters=64, kernel_size=5, padding='same')
        self.batchnorm3 = BatchNormalization()
        self.leakyrelu3 = LeakyReLU()

        self.flatten = Flatten()

        self.dense1 = Dense(2048)
        self.batchnorm4 = BatchNormalization()
        self.leakyrelu4 = LeakyReLU()

        self.dense2 = Dense(100)


    def call(self, image):
        """ Passes input image through the network. """

        x = self.conv1(image)
        x = self.leakyrelu1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.leakyrelu2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.leakyrelu3(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.batchnorm4(x)
        x = self.leakyrelu4(x)

        x = self.dense2(x)

        return x

    def loss_function(self, y_true, y_pred):

        # test to ensure axes are correct
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true)))
