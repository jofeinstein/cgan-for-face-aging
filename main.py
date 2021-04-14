import numpy as np
import tensorflow as tf
from load_data import *
from encoder_model import Encoder
from discriminator_model import Discriminator
from generator_model import Generator
from fr_model import FaceRecognition
import argparse

## load in data

## build and compile generator, discriminator, adversarial networks

## 3 training steps:


def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-data_dir_path',
                        default='imdb_crop/',
                        required=False)
    parser.add_argument('-mat_file_path',
                        default='imdb_crop/imdb.mat',
                        required=False)
    parser.add_argument('-num_classes',
                        default=6,
                        required=False)
    return parser.parse_args()


args = getArgs()


def main():
    # Load in data
    # full_image_path_list, label_list = load_meta_data(data_dir_path=args.data_dir_path, mat_file_path=args.mat_file_path)
    # image_list = load_images_and_labels(image_path_list=full_image_path_list)
    # label_one_hot = tf.one_hot(np.reshape(np.asarray(label_list), newshape=(len(label_list))), depth=args.num_classes)

    # compile generator and discriminator

    generator = Generator()
    generator(tf.keras.Input(shape=(100,)), tf.keras.Input(shape=(6,)))
    generator.compile(optimizer=generator.optimizer, loss=generator.loss_function)

    discriminator = Discriminator()
    discriminator(tf.keras.Input(shape=(64, 64, 3)), tf.keras.Input(shape=(6,)))
    discriminator.compile(optimizer=discriminator.optimizer, loss=discriminator.loss_function)


main()
