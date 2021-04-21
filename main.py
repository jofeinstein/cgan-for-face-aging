import numpy as np
import tensorflow as tf
from load_data import *
from encoder_model import Encoder
from discriminator_model import Discriminator
from generator_model import Generator
from fr_model import FaceRecognition
import argparse
import os
import time
from matplotlib import pyplot as plt
from tensorflow.keras.applications import ResNet50V2

print("\nNum GPUs Available: {}\n".format(len(tf.config.list_physical_devices('GPU'))))
# tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-data_dir_path',
                        default=os.getcwd() + '/data/imdb_crop/',
                        help='path to directory containing training data',
                        required=False)
    parser.add_argument('-mat_file_path',
                        default=os.getcwd() + '/data/imdb_crop/imdb.mat',
                        help='path to mat file downloaded from dataset',
                        required=False)
    parser.add_argument('-num_epochs',
                        default=25,
                        help='number of epochs to train for',
                        type=int,
                        required=False)
    parser.add_argument('-batch_size',
                        default=256,
                        type=int,
                        required=False)
    parser.add_argument('-save_dir',
                        default=os.getcwd() + '/data/',
                        help='path to directory to save weights and images to',
                        required=False)
    parser.add_argument('-encoder_train_size',
                        default=2500,
                        type=int,
                        help='number of examples to train encoder on',
                        required=False)
    parser.add_argument('-phase',
                        help='phase of the training process',
                        choices=['cgan', 'encoder', 'optimization'],
                        default=None,
                        required=True)
    parser.add_argument('-num_images',
                        help='how many images to train on',
                        default=100000,
                        type=int,
                        required=False)
    return parser.parse_args()


# global variables
args = getArgs()
latent_dim = 100
num_classes = 6
image_shape = (64, 64, 3)


def plot_list(lst, title):
    fig = plt.figure()
    plt.plot(lst)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.title(title)
    # plt.legend()
    plt.draw()
    fig.savefig(args.save_dir + 'training_logs/' + title + '.png', dpi=500)


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


def compile_generator():
    """
    Compiles generator and discriminator models

    :return: tuple of compiled generator and discriminator models
    """
    generator = Generator()
    generator((tf.keras.Input(shape=(latent_dim,)), tf.keras.Input(shape=(num_classes,))))
    generator.compile(optimizer=generator.optimizer, loss='binary_crossentropy')

    return generator


def compile_discriminator():
    discriminator = Discriminator()
    discriminator((tf.keras.Input(shape=image_shape), tf.keras.Input(shape=(num_classes,))))
    discriminator.compile(optimizer=discriminator.optimizer, loss='binary_crossentropy')

    return discriminator


def compile_encoder():
    encoder = Encoder()
    encoder(tf.keras.Input(shape=image_shape))
    encoder.compile(optimizer=encoder.optimizer, loss='binary_crossentropy')

    return encoder


def euclidean_distance_loss(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))


def compile_fr(generator, encoder):
    image = tf.keras.Input(shape=image_shape)
    label = tf.keras.Input(shape=(num_classes,))
    fr_model_optimizer = tf.keras.optimizers.Adam()
    fr_adversarial_optimizer = tf.keras.optimizers.Adam()

    resent_model = ResNet50V2(include_top=False, weights='imagenet')
    image_input = resent_model.input
    x = resent_model.layers[-1].output
    x = tf.keras.layers.Dense(128)(x)
    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(x)
    fr_model = tf.keras.Model([image_input], output)

    fr_model.compile(optimizer=fr_model_optimizer, loss='binary_crossentropy')

    fr_model.trainable = False
    latent_vector = encoder(image)
    gen_images = generator([latent_vector, label])

    resized = tf.keras.layers.experimental.preprocessing.Resizing(height=224, width=224)(gen_images)

    embeddings = fr_model(resized)
    fr_adversarial = tf.keras.Model([image, label], embeddings)
    fr_adversarial.compile(optimizer=fr_adversarial_optimizer, loss=euclidean_distance_loss)

    return fr_model, fr_adversarial


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

    :param image_array: 4d array containing all training images. shape (batch, height, width, channels)
    :param label_one_hot: 2d array containing one hot vectors that correspond to labels of images in image_array
    :param generator: compiled generator model
    :param discriminator: compiled discriminator model
    :param cgan: compiled cGAN model
    :return: A crisp high five
    """

    print("\nTraining cGAN...\n")

    d_loss1 = []
    d_loss2 = []
    cgan_loss_lst = []

    for epoch in range(args.num_epochs):
        d_batch_loss1 = []
        d_batch_loss2 = []
        cgan_loss_batch_lst = []
        start_time = time.time()

        for x in range(0, len(image_array), args.batch_size):
            batch_images = image_array[x: x + args.batch_size]
            batch_labels = label_one_hot[x: x + args.batch_size]
            noise1, noise2, f_labels_one_hot = gen_fake_data(len(batch_labels))
            true_labels = np.ones((len(batch_labels), 1))
            fake_labels = np.zeros((len(batch_labels), 1))

            # training discriminator
            gen_images = generator.predict_on_batch([noise1, batch_labels])

            discriminator_loss_real = discriminator.train_on_batch([batch_images, batch_labels], true_labels)
            discriminator_loss_fake = discriminator.train_on_batch([gen_images, batch_labels], fake_labels)

            d_batch_loss1.append(discriminator_loss_real)
            d_batch_loss2.append(discriminator_loss_fake)

            # training generator
            cgan_loss = cgan.train_on_batch([noise2, f_labels_one_hot], true_labels)
            cgan_loss_batch_lst.append(cgan_loss)

        d_loss1.append(np.mean(d_batch_loss1))
        d_loss2.append(np.mean(d_batch_loss2))
        cgan_loss_lst.append(np.mean(cgan_loss_batch_lst))

        # generate 5 test images every 5 epochs and save
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.num_epochs:
            noise1 = np.random.normal(0, 1, size=(5, latent_dim))
            gen_images = generator.predict_on_batch([noise1, label_one_hot[0:5]])

            for i in range(gen_images.shape[0]):
                dirr = args.save_dir + 'training_imgs/epoch' + str(epoch) + '/'
                if not os.path.exists(dirr):
                    os.makedirs(dirr)

                img_array = gen_images[i]
                img = Image.fromarray(((img_array * 255).astype(np.uint8)))
                img.save(dirr + str(i) + 'test.png')

        # save weights every 50 epochs
        if (epoch + 1) % 50 == 0:
            generator.save(args.save_dir + "weights/generator_checkpoint{}".format(str(epoch)), save_format="tf")
            discriminator.save(args.save_dir + "weights/discriminator_checkpoint{}".format(str(epoch)), save_format="tf")

        avg_d_loss = (np.mean(d_batch_loss1) + np.mean(d_batch_loss2)) / 2
        print("Epoch: {} / {}        Discriminator Loss: {}      cGAN Loss: {}      Time Elapsed: {}s".format(epoch + 1, args.num_epochs,
                                                                                      avg_d_loss,
                                                                                      np.mean(cgan_loss_batch_lst), round(time.time() - start_time, 2)))

    # save weights and losses
    if not os.path.exists(args.save_dir + 'weights'):
        os.makedirs(args.save_dir + 'weights')
    generator.save_weights(args.save_dir + "weights/generator.h5")
    discriminator.save_weights(args.save_dir + "weights/discriminator.h5")

    out_dl1 = open(args.save_dir + "training_logs/discriminator_loss1.txt", 'w')
    out_dl2 = open(args.save_dir + "training_logs/discriminator_loss2.txt", 'w')
    out_cganl = open(args.save_dir + "training_logs/cgan_loss.txt", 'w')

    print(d_loss1, file=out_dl1)
    print(d_loss2, file=out_dl2)
    print(cgan_loss_lst, file=out_cganl)

    # create graphs of losses
    plot_list(cgan_loss_lst, "generator_loss")
    avg_discriminator_loss_lst = []
    for i in range(len(d_loss1)):
        avg_discriminator_loss_lst.append(np.mean([d_loss1[i], d_loss2[i]]))
    plot_list(avg_discriminator_loss_lst, "discriminator_loss")

    print("I'm so tired... let me sleep...")


def encoder_training(encoder, generator):
    """
    Training loop for encoder. Must be run after initial cGAN training.
    :param generator: compiled generator model
    :return: sadness
    """

    print("\nTraining encoder...\n")

    # load generator weights

    try:
        generator.load_weights(args.save_dir + "weights/generator.h5")
        # generator = tf.keras.models.load_model(args.save_dir + "weights/generator_checkpoint99")
    except OSError:
        print("Error: Could not find weights for generator. Ensure weights are stored in data/weights/generator.h5")
        return

    # create random labels and latent vectors for training
    r_labels = np.random.randint(0, num_classes, args.encoder_train_size)
    r_labels_one_hot = tf.one_hot(np.asarray(r_labels), depth=num_classes)
    r_latent_v = np.random.normal(0, 1, size=(args.encoder_train_size, latent_dim))

    encoder_loss_lst = []
    for epoch in range(args.num_epochs):
        encoder_loss_batch_lst = []
        start_time = time.time()

        for x in range(0, args.encoder_train_size, args.batch_size):
            batch_latent_v = r_latent_v[x: x + args.batch_size]
            batch_labels = r_labels_one_hot[x: x + args.batch_size]

            # generate images using random latent vectors
            gen_images = generator.predict_on_batch([batch_latent_v, batch_labels])

            # train encoder to encode generated images to provided latent vectors
            encoder_loss = encoder.train_on_batch(gen_images, batch_latent_v)
            encoder_loss_batch_lst.append(encoder_loss)

        encoder_loss_lst.append(np.mean(encoder_loss_batch_lst))

        # save weights every 50 epochs
        if (epoch + 1) % 50 == 0:
            encoder.save(args.save_dir + "weights/encoder_checkpoint{}".format(str(epoch)), save_format="tf")

        print("Epoch: {} / {}        Encoder Loss: {}       Time Elapsed: {}s".format(epoch + 1, args.num_epochs,
                                                                                      np.mean(encoder_loss_batch_lst),
                                                                                      time.time() - start_time))

    # save encoder weight and losses
    if not os.path.exists(args.save_dir + 'weights'):
        os.makedirs(args.save_dir + 'weights')
    encoder.save_weights(args.save_dir + "weights/encoder.h5")

    out_el = open(args.save_dir + "training_logs/encoder_loss.txt", 'w')
    print(encoder_loss_lst, file=out_el)

    # graph losses
    plot_list(encoder_loss_lst, "encoder_loss")


def fr_optimzation_training(image_array, label_one_hot, generator, encoder):
    # compile fr models
    fr_model, fr_adversarial = compile_fr(generator, encoder)

    image_input = tf.keras.Input(shape=(64, 64, 3))
    resized_image = tf.keras.layers.experimental.preprocessing.Resizing(height=224, width=224)(image_input)
    resizer = tf.keras.Model(image_input, resized_image)
    resizer.compile()

    # load encoder and generator weights
    try:
        encoder.load_weights(args.save_dir + "weights/encoder.h5")
        generator.load_weights(args.save_dir + "weights/generator.h5")
    except OSError:
        print("Error: Could not find weights for generator or encoder. Ensure weights are stored in data/weights/")
        return

    loss_lst = []
    for epoch in range(args.num_epochs):

        start_time = time.time()
        batch_loss = []
        for x in range(0, len(image_array), args.batch_size):
            batch_images = image_array[x: x + args.batch_size]
            batch_labels = label_one_hot[x: x + args.batch_size]

            resized_images = resizer.predict_on_batch(batch_images)
            embeddings = fr_model.predict_on_batch(resized_images)

            loss = fr_adversarial.train_on_batch([batch_images, batch_labels], embeddings)
            batch_loss.append(loss)

        loss_lst.append(np.mean(batch_loss))

        # generate test images every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.num_epochs:
            noise1 = np.random.normal(0, 1, size=(5, latent_dim))
            gen_images = generator.predict_on_batch([noise1, label_one_hot[0:5]])

            for i in range(gen_images.shape[0]):
                dirr = args.save_dir + 'training_imgs_optimization/epoch' + str(epoch) + '/'
                if not os.path.exists(dirr):
                    os.makedirs(dirr)

                img_array = gen_images[i]
                img = Image.fromarray(((img_array * 255).astype(np.uint8)))
                img.save(dirr + str(i) + '_opt_test.png')

        # save weights every 50 epochs
        if (epoch + 1) % 50 == 0:
            generator.save(args.save_dir + "weights/generator_opt_checkpoint{}".format(str(epoch)), save_format="tf")
            encoder.save(args.save_dir + "weights/discriminator_opt_checkpoint{}".format(str(epoch)), save_format="tf")

        print("Epoch: {} / {}        Encoder Loss: {}       Time Elapsed: {}s".format(epoch + 1, args.num_epochs,
                                                                                      np.mean(batch_loss),
                                                                                      time.time() - start_time))

    # save weight and losses
    if not os.path.exists(args.save_dir + 'weights'):
        os.makedirs(args.save_dir + 'weights')
    encoder.save_weights(args.save_dir + "weights/encoder_opt.h5")
    generator.save_weights(args.save_dir + "weights/generator_opt.h5")

    out_el = open(args.save_dir + "training_logs/reconstruction_loss.txt", 'w')
    print(loss_lst, file=out_el)

    # graph losses
    plot_list(loss_lst, "reconstruction_loss")

    print("owwww.")


def main():
    # from matplotlib import pyplot as plt
    # for i in range(5):
    #     print(label_one_hot[i], full_image_path_list[i])
    #     plt.imshow(image_array[i], interpolation='nearest')
    #     plt.show()

    # compile generator, discriminator, cgan

    if args.phase == 'cgan':
        # compile models
        generator = compile_generator()
        discriminator = compile_discriminator()
        cgan = compile_cgan(generator, discriminator)

        # Load in data
        full_image_path_list, label_one_hot = load_meta_data(args.data_dir_path, args.mat_file_path, args.num_images)
        image_array = load_images(full_image_path_list, (64, 64))

        cgan_training(image_array, label_one_hot, generator, discriminator, cgan)

    elif args.phase == 'encoder':
        # compile models
        encoder = compile_encoder()
        generator = compile_generator()

        encoder_training(encoder, generator)

    elif args.phase == 'optimization':
        # compile models
        generator = compile_generator()
        encoder = compile_encoder()

        # Load data
        full_image_path_list, label_one_hot = load_meta_data(args.data_dir_path, args.mat_file_path, args.num_images)
        image_array = load_images(full_image_path_list, (64, 64))

        fr_optimzation_training(image_array, label_one_hot, generator, encoder)


main()
