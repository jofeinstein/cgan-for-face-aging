import os
import scipy.io
from datetime import datetime
from PIL import Image
import numpy as np
import tensorflow as tf


def load_meta_data(data_dir_path, mat_file_path, image_num=100000):
    """
    Loads paths of all images and the corresponding labels from matlab file

    :param data_dir_path: path to the data directory
    :param mat_file_path: path to the matlab meta information file
    :param image_num: number of images to use from dataset
    :return: list of image paths and list of corresponding labels
    """

    print("Loading meta information...")

    # retrieving path, birthday, and photo data info from matlab file
    mat_file = scipy.io.loadmat(mat_file_path)
    relative_image_path_list = mat_file['imdb'][0, 0]["full_path"][0]

    full_image_path_list = [data_dir_path + x[0] for x in relative_image_path_list]

    birthday = mat_file['imdb'][0, 0]["dob"][0]
    photo_taken_year = mat_file['imdb'][0, 0]["photo_taken"][0]

    label_list = []

    # converting birthday and photo date to age labels
    for i, path in enumerate(full_image_path_list):
        age = photo_taken_year[i] - datetime.fromordinal(birthday[i]).year
        label_list.append(convert_age_to_label(age))

    # randomly selecting image_num images from dataset
    random_indices = np.random.choice(np.arange(len(label_list)), image_num, replace=False)
    full_image_path_list = list(np.asarray(full_image_path_list)[random_indices])
    label_array = np.asarray(label_list)[random_indices]

    # creating one hot of label_array
    label_one_hot = tf.one_hot(label_array, depth=len(set(label_list)))

    return full_image_path_list, label_one_hot


def convert_age_to_label(age):
    """
    Converts age to one of six labels
    :param age: an int representing a person's age
    :return: the corresponding label
    """
    if 0 <= age <= 18:
        label = 0
    elif 19 <= age <= 29:
        label = 1
    elif 30 <= age <= 39:
        label = 2
    elif 40 <= age <= 49:
        label = 3
    elif 50 <= age <= 59:
        label = 4
    else:
        label = 5

    return label


def load_images(image_path_list, image_size=(64, 64)):
    """
    Loads and resizes all images in dataset

    :param image_path_list: list of all image paths
    :param image_size: size tuple (width, height) to resize images to
    :return: numpy array of size (num_images, width, height) containing all images
    """
    print('Loading images into array...')

    image_list = []

    for i, image_path in enumerate(image_path_list):
        if i % 1000 == 0:
            print('{} / {}'.format(i, len(image_path_list)))
        img = Image.open(image_path)
        img = img.resize(image_size)
        img = np.array(img, dtype=np.float32)
        img /= 255.

        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        image_list.append(img)

    image_array = np.stack(image_list, axis=0)

    return image_array
