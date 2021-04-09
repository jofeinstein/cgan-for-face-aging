import os
import scipy.io
from datetime import datetime


def load_meta_data(data_dir_path, mat_file_path):

    mat_file = scipy.io.loadmat(mat_file_path)
    relative_image_path_list = mat_file['wiki'][0, 0]["full_path"][0]

    full_image_path_list = [data_dir_path + x[0] for x in relative_image_path_list]

    birthday = mat_file['wiki'][0, 0]["dob"][0]
    photo_taken_year = mat_file['wiki'][0, 0]["photo_taken"][0]

    path_age_tuple_list = []

    for i, path in enumerate(full_image_path_list):
        age = photo_taken_year[i] - datetime.fromordinal(birthday[i]).year
        path_age_tuple_list.append((path, age))

    return path_age_tuple_list
#
#
# lst = load_meta_data('/Users/jofeinstein/Downloads/wiki/' ,'/Users/jofeinstein/Downloads/wiki/wiki.mat')
# print(lst[0])