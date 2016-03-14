import os
import csv
import json
import numpy as np

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
DATA_FOLDER = os.path.join(HERE_PATH, '..', 'data')


# adding parent directory to path, so we can access the utils easily
import sys
utils_path = os.path.join(HERE_PATH, '..')
sys.path.append(utils_path)

# filetools is a custom library that we provide here
# this library can be found online at https://github.com/jgrizou/filetools
from utils import filetools

X_FILENAME_PATTERN = 'params.json'
Y_FILENAME_PATTERN = 'features.json'
VIDEO_FILENAME = 'video.avi'


def load_json_file(filename):
    with open(filename) as f:
        return json.load(f)


def save_json_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def save_numpy_to_csv(data, filename, header=None):
    np.savetxt(filename, data, header=header)


def save_list_to_csv(data_list, filename):
    with open(filename, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        for data in data_list:
            csvwriter.writerow(data)


def get_folder_info(foldername):

    x_filename = os.path.join(foldername, X_FILENAME_PATTERN)
    x_data = load_json_file(x_filename)

    y_filename = os.path.join(foldername, Y_FILENAME_PATTERN)
    y_data = load_json_file(y_filename)

    folder_info = {}
    folder_info['path'] = os.path.relpath(foldername, DATA_FOLDER)
    folder_info['x'] = x_data
    folder_info['y'] = y_data

    return folder_info


def gather_dataset_info(foldername):

    # list and sort all folder
    folder_list = filetools.list_folders(foldername)
    folder_list.sort()

    dataset_info = []
    for folder in folder_list:
        dataset_info.append(get_folder_info(folder))

    return dataset_info


def save_dataset_info(dataset_info, foldername):

    # we assume all dataset info are consistent, i.e. they all have the same param and feature names
    count = len(dataset_info)
    x_keys = dataset_info[0]['x'].keys()
    y_keys = dataset_info[0]['y'].keys()

    # save basic info
    info = {}
    info['count'] = count
    info['x_keys'] = x_keys
    info['y_keys'] = y_keys

    info_filename = os.path.join(foldername, 'info.json')
    save_json_to_file(info, info_filename)

    # experimental parameters, numpy compatible
    x = np.zeros((count, len(x_keys)))
    for i in range(count):
        for j, k in enumerate(x_keys):
            x[i, j] = dataset_info[i]['x'][k]

    x_filename = os.path.join(foldername, 'x.csv')
    x_header = ' '.join(x_keys)
    save_numpy_to_csv(x, x_filename, x_header)

    # observations, experimental ouputs, numpy compatible
    y = np.zeros((count, len(y_keys)))
    for i in range(count):
        for j, k in enumerate(y_keys):
            y[i, j] = dataset_info[i]['y'][k]

    y_filename = os.path.join(foldername, 'y.csv')
    y_header = ' '.join(y_keys)
    save_numpy_to_csv(y, y_filename, y_header)

    # corresponding folder
    path = []
    for i in range(count):
        path.append(os.path.normpath(dataset_info[i]['path']))

    path_filename = os.path.join(foldername, 'path.json')
    save_json_to_file(path, path_filename)

    # all data summarized in one csv
    data_list = []
    column_names = []
    column_names.append('path')
    column_names.extend(x_keys)
    column_names.extend(y_keys)
    data_list.append(column_names)
    for i in range(count):
        data = []
        data.append(path[i])
        data.extend(list(x[i, :]))
        data.extend(list(y[i, :]))
        data_list.append(data)

    full_filename = os.path.join(foldername, 'full.csv')
    save_list_to_csv(data_list, full_filename)


def generate_folder_name(dataset_info):

    fields = dataset_info[0]['x'].keys()
    fields.extend(dataset_info[0]['y'].keys())

    return '_'.join(fields)


if __name__ == "__main__":

    dataset_folders = filetools.list_folders(DATA_FOLDER)

    for folder in dataset_folders:

        dataset_info = gather_dataset_info(folder)

        dataset_foldername = generate_folder_name(dataset_info)
        filetools.ensure_dir(dataset_foldername)

        save_dataset_info(dataset_info, dataset_foldername)
