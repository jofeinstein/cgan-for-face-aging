import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50V2

class FaceRecognition(tf.keras.Model):
    def __init__(self):

        super(FaceRecognition, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam()
        self.resnet_pretrained = ResNet50V2(include_top=True, weights='imagenet', input_shape=(96, 96, 3))
        self.fc = Dense(128)

    def call(self, image):
        """ Passes input image through the network. """

        x = self.resnet_pretrained.layers[-1].output
        x = self.fc(x)
        x = tf.math.l2_normalize(x, axis=-1)

        return x

