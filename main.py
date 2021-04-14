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
    parser.add_argument('-epoch',
                        default=25,
                        required=False)
    parser.add_argument('-batch_size',
                        default=256,
                        required=False)
    return parser.parse_args()


args = getArgs()
latent_dim = 100
num_classes = 6
image_shape = (64, 64, 3)


def compile_gan(generator_model, discriminator_model):
    discriminator_model.trainable = False
    noise = tf.keras.Input((latent_dim,))
    label = tf.keras.Input((num_classes,))
    reconstructed_image = generator_model((noise, label))

    out_label = discriminator_model((reconstructed_image, label))
    # define gan model as taking noise and label and outputting a classification
    model = tf.keras.Model([noise, label], out_label)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')

    return model


def compile_gen_disc():
    generator = Generator()
    generator((tf.keras.Input(shape=(latent_dim,)), tf.keras.Input(shape=(num_classes,))))
    generator.compile(optimizer=generator.optimizer, loss='binary_crossentropy')

    discriminator = Discriminator()
    discriminator((tf.keras.Input(shape=image_shape), tf.keras.Input(shape=(num_classes,))))
    discriminator.compile(optimizer=discriminator.optimizer, loss='binary_crossentropy')

    return generator, discriminator


def gen_fake_data():
    noise1 = np.random.normal(0, 1, size=(args.batch_size, latent_dim))
    noise2 = np.random.normal(0, 1, size=(args.batch_size, latent_dim))

    f_labels = np.random.randint(0, num_classes-1, args.batch_size)
    f_labels_one_hot = tf.one_hot(np.asarray(f_labels), depth=num_classes)

    return noise1, noise2, f_labels_one_hot


def main():
    # Load in data
    full_image_path_list, label_list = load_meta_data(data_dir_path=args.data_dir_path, mat_file_path=args.mat_file_path)
    image_array = load_images_and_labels(image_path_list=full_image_path_list)
    label_one_hot = tf.one_hot(np.asarray(label_list), depth=num_classes)

    # compile generator, discriminator, gan
    generator, discriminator = compile_gen_disc()
    gan = compile_gan(generator, discriminator)

    true_labels = np.ones((args.batch_size, 1))
    fake_labels = np.zeros((args.batch_size, 1))

    # initial gan training
    for epoch in range(args.epoch):
        for x in range(0, len(label_list), args.batch_size):
            batch_images = image_array[x: x + args.batch_size]
            batch_labels = label_one_hot[x: x + args.batch_size]
            noise1, noise2, f_labels_one_hot = gen_fake_data()

            # training discriminator
            gen_images = generator.predict_on_batch([noise1, batch_labels])

            d_loss_real = discriminator.train_on_batch([batch_images, batch_labels], true_labels)
            d_loss_fake = discriminator.train_on_batch([gen_images, batch_labels], fake_labels)

            # training generator
            gan_loss = gan.train_on_batch([noise2, f_labels_one_hot], true_labels)


main()
