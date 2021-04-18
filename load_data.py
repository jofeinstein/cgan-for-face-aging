import os
import scipy.io
from datetime import datetime
from PIL import Image
import numpy as np
import tensorflow as tf


def load_meta_data(data_dir_path, mat_file_path, num_images):
    """
    Loads paths of all images and the corresponding labels from matlab file

    :param data_dir_path: path to the data directory
    :param mat_file_path: path to the matlab meta information file
    :param num_images: number of images to use from dataset
    :return: list of image paths and list of corresponding labels
    """

    print("Loading meta information...")

    # retrieving path, birthday, and photo data info from matlab file
    mat_file = scipy.io.loadmat(mat_file_path)
    relative_image_path_list = mat_file['imdb'][0, 0]["full_path"][0]

    full_image_path_list = [data_dir_path + x[0] for x in relative_image_path_list]

    birthday = mat_file['imdb'][0, 0]["dob"][0]
    photo_taken_year = mat_file['imdb'][0, 0]["photo_taken"][0]

    cat0_list = []
    cat1_list = []
    cat2_list = []
    cat3_list = []
    cat4_list = []
    cat5_list = []

    # converting birthday and photo date to age labels
    # ensuring that the training data is as balanced as possible
    for i, path in enumerate(full_image_path_list):
        age = photo_taken_year[i] - datetime.fromordinal(birthday[i]).year
        label = convert_age_to_label(age)

        if label == 0:
            cat0_list.append([path, label])
        elif label == 1:
            cat1_list.append([path, label])
        elif label == 2:
            cat2_list.append([path, label])
        elif label == 3:
            cat3_list.append([path, label])
        elif label == 4:
            cat4_list.append([path, label])
        elif label == 5:
            cat5_list.append([path, label])

    # randomly selecting image_num images from dataset
    random_indices0 = np.random.choice(np.arange(len(cat0_list)), int(np.ceil(num_images / 6)), replace=False)
    random_indices1 = np.random.choice(np.arange(len(cat1_list)), int(np.ceil(num_images / 6)), replace=False)
    random_indices2 = np.random.choice(np.arange(len(cat2_list)), int(np.ceil(num_images / 6)), replace=False)
    random_indices3 = np.random.choice(np.arange(len(cat3_list)), int(np.ceil(num_images / 6)), replace=False)
    random_indices4 = np.random.choice(np.arange(len(cat4_list)), int(np.ceil(num_images / 6)), replace=False)
    random_indices5 = np.random.choice(np.arange(len(cat5_list)), int(np.ceil(num_images / 6)), replace=False)

    cat0_random = np.asarray(cat0_list)[random_indices0]
    cat1_random = np.asarray(cat1_list)[random_indices1]
    cat2_random = np.asarray(cat2_list)[random_indices2]
    cat3_random = np.asarray(cat3_list)[random_indices3]
    cat4_random = np.asarray(cat4_list)[random_indices4]
    cat5_random = np.asarray(cat5_list)[random_indices5]

    full_path_label_array = np.vstack([cat0_random, cat1_random, cat2_random, cat3_random, cat4_random, cat5_random])

    label_array = full_path_label_array[:, 1]
    new_image_path_list = list(full_path_label_array[:, 0])

    # creating one hot of label_array
    label_one_hot = tf.one_hot(label_array, depth=6)

    return new_image_path_list, label_one_hot


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
    # print('Loading images into array...')

    image_list = []

    for i, image_path in enumerate(image_path_list):
        # if i % 1000 == 0:
        #     print('{} / {}'.format(i, len(image_path_list)))
        img = Image.open(image_path)
        img = img.resize(image_size)
        img = np.array(img, dtype=np.float32)
        img /= 255.

        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        image_list.append(img)

    image_array = np.stack(image_list, axis=0)

    return image_array
