import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50V2

class FaceRecognition(tf.keras.Model):
    def __init__(self):

        super(FaceRecognition, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam()

        self.resnet_pretrained = ResNet50V2(include_top=True, weights='imagenet')

        self.fc = Dense(128)

    def call(self, image):
        """ Passes input image through the network. """

        x = self.resent_model(image)
        x = self.fc(x)

        return x

    def loss_function(self, y_true, y_pred):
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        return loss_fn(y_true, y_pred)

