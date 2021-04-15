import numpy as np
import tensorflow as tf
from load_data import *
from encoder_model import Encoder
from discriminator_model import Discriminator
from generator_model import Generator
from fr_model import FaceRecognition
import argparse
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)


def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-data_dir_path',
                        default='imdb_crop/',
                        help='path to directory containing training data',
                        required=False)
    parser.add_argument('-mat_file_path',
                        default='imdb_crop/imdb.mat',
                        help='path to mat file downloaded from dataset',
                        required=False)
    parser.add_argument('-num_epochs',
                        default=25,
                        help='number of epochs to train for',
                        required=False)
    parser.add_argument('-batch_size',
                        default=256,
                        required=False)
    parser.add_argument('-save_dir',
                        default='',
                        help='path to directory to save weights and images to',
                        required=False)
    parser.add_argument('-encoder_train_size',
                        default=2500,
                        help='number of examples to train encoder on',
                        required=False)
    parser.add_argument('-training_step',
                        help='which step of training process. [initial_cgan, encoder, optimization]',
                        default=None,
                        required=True)
    return parser.parse_args()


# global variables
args = getArgs()
latent_dim = 100
num_classes = 6
image_shape = (64, 64, 3)


def compile_cgan(generator_model, discriminator_model):
    """
    Compiles the cGAN model using already compiled generator and discriminator models

    :param generator_model: compiled generator model
    :param discriminator_model: compiled discriminator model
    :return: compiled cGAN model
    """
    discriminator_model.trainable = False
    noise = tf.keras.Input((latent_dim,))
    label = tf.keras.Input((num_classes,))
    reconstructed_image = generator_model((noise, label))

    out_label = discriminator_model((reconstructed_image, label))

    model = tf.keras.Model([noise, label], out_label)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')

    return model


def compile_gen_disc():
    """
    Compiles generator and discriminator models

    :return: tuple of compiled generator and discriminator models
    """
    generator = Generator()
    generator((tf.keras.Input(shape=(latent_dim,)), tf.keras.Input(shape=(num_classes,))))
    generator.compile(optimizer=generator.optimizer, loss='binary_crossentropy')

    discriminator = Discriminator()
    discriminator((tf.keras.Input(shape=image_shape), tf.keras.Input(shape=(num_classes,))))
    discriminator.compile(optimizer=discriminator.optimizer, loss='binary_crossentropy')

    return generator, discriminator


def gen_fake_data(num_ex=args.batch_size):
    """
    Generates fake data for use in cGAN training.

    :param num_ex: number of fake examples to generate
    :return: two different noise vectors of shape (num_ex, latent_dim) and a matrix of
             random one hot labels of shape (num_classes, num_ex)
    """
    noise1 = np.random.normal(0, 1, size=(num_ex, latent_dim))
    noise2 = np.random.normal(0, 1, size=(num_ex, latent_dim))

    f_labels = np.random.randint(0, num_classes, num_ex)
    f_labels_one_hot = tf.one_hot(np.asarray(f_labels), depth=num_classes)

    return noise1, noise2, f_labels_one_hot


def cgan_training(image_array, label_one_hot, generator, discriminator, cgan):
    """
    Training loop for initial cGAN training. Must be trained before other optimizations.

    :param image_array: 4d array containing all training images of shape (num_images, width, height, channels)
    :param label_one_hot: 2d array containing one hot vectors that correspond to labels of images in image_array
    :param generator: compiled generator model
    :param discriminator: compiled discriminator model
    :param cgan: compiled cGAN model
    :return: A crisp high five
    """

    print("Training cGAN...")

    true_labels = np.ones((args.batch_size, 1))
    fake_labels = np.zeros((args.batch_size, 1))

    d_loss_real_lst = []
    d_loss_fake_lst = []
    cgan_loss_lst = []

    for epoch in range(args.num_epochs):
        d_loss_real_batch_lst = []
        d_loss_fake_batch_lst = []
        cgan_loss_batch_lst = []

        for x in range(0, len(image_array), args.batch_size):
            batch_images = image_array[x: x + args.batch_size]
            batch_labels = label_one_hot[x: x + args.batch_size]
            noise1, noise2, f_labels_one_hot = gen_fake_data()

            # training discriminator
            gen_images = generator.predict_on_batch([noise1, batch_labels])

            discriminator_loss_real = discriminator.train_on_batch([batch_images, batch_labels], true_labels)
            discriminator_loss_fake = discriminator.train_on_batch([gen_images, batch_labels], fake_labels)

            d_loss_real_batch_lst.append(discriminator_loss_real)
            d_loss_fake_batch_lst.append(discriminator_loss_fake)

            # training generator
            cgan_loss = cgan.train_on_batch([noise2, f_labels_one_hot], true_labels)
            cgan_loss_batch_lst.append(cgan_loss)

        d_loss_real_lst.append(np.mean(d_loss_real_batch_lst))
        d_loss_fake_lst.append(np.mean(d_loss_fake_batch_lst))
        cgan_loss_lst.append(np.mean(cgan_loss_batch_lst))

        # generate 5 test images every 5 epochs and save
        if (epoch + 1) % 5 == 0:
            noise1, noise2, f_labels_one_hot = gen_fake_data(num_ex=5)
            gen_images = generator.predict_on_batch([noise1, f_labels_one_hot])

            for i, img_array in enumerate(gen_images):
                dirr = args.save_dir + '/epoch' + str(epoch) + '/'
                if not os.path.exists(dirr):
                    os.makedirs(dirr)

                img = Image.fromarray(img_array)
                img.save(dirr + str(i) + 'test.png')

        avg_d_loss = (np.mean(d_loss_real_batch_lst) + np.mean(d_loss_fake_batch_lst)) / 2
        print("Epoch: {} / {}        Discriminator Loss: {}      cGAN Loss: {}".format(epoch + 1, args.num_epochs,
                                                                                      avg_d_loss,
                                                                                      np.mean(cgan_loss_batch_lst)))

    # save weights
    generator.save_weights("generator.h5")
    discriminator.save_weights("discriminator.h5")


def encoder_training(generator):
    """
    Training loop for encoder. Must be run after initial cGAN training.
    :param generator: compiled generator model
    :return: sadness
    """

    print("Training encoder...")

    # compile encoder and load generator weights
    encoder = Encoder()
    encoder(tf.keras.Input((64, 64, 3)))
    encoder.compile(optimizer=encoder.optimizer, loss='binary_crossentropy')
    generator.load_weights("generator.h5")

    # create random labels and latent vectors for training
    r_labels = np.random.randint(0, num_classes, args.encoder_train_size)
    r_labels_one_hot = tf.one_hot(np.asarray(r_labels), depth=num_classes)
    r_latent_v = np.random.normal(0, 1, size=(args.encoder_train_size, latent_dim))

    encoder_loss_lst = []
    for epoch in range(args.num_epochs):
        encoder_loss_batch_lst = []

        for x in range(0, args.encoder_train_size, args.batch_size):
            batch_latent_v = r_latent_v[x: x + args.batch_size]
            batch_labels = r_labels_one_hot[x: x + args.batch_size]

            # generate images using random latent vectors
            gen_images = generator.predict_on_batch([batch_latent_v, batch_labels])

            # train encoder to encode generated images to provided latent vectors
            encoder_loss = encoder.train_on_batch(gen_images, batch_latent_v)
            encoder_loss_batch_lst.append(encoder_loss)

        encoder_loss_lst.append(np.mean(encoder_loss_batch_lst))
        print("Epoch: {} / {}        Encoder Loss: {}".format(epoch + 1, args.num_epochs,
                                                              np.mean(encoder_loss_batch_lst)))

    # save encoder weight
    encoder.save_weights("encoder.h5")


def main():
    # Load in data
    full_image_path_list, label_list = load_meta_data(args.data_dir_path, args.mat_file_path)
    image_array = load_images_and_labels(full_image_path_list)
    label_one_hot = tf.one_hot(np.asarray(label_list), depth=num_classes)

    # compile generator, discriminator, cgan
    generator, discriminator = compile_gen_disc()
    cgan = compile_cgan(generator, discriminator)

    if args.training_step == 'initial_cgan':
        cgan_training(image_array, label_one_hot, generator, discriminator, cgan)

    elif args.training_step == 'encoder':
        encoder_training(generator)


main()
