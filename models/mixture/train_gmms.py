import os
import csv
import pickle
import numpy as np

from sklearn.mixture import GMM

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..', '..')
sys.path.append(root_path)

from utils import filetools
from datasets.tools import load_dataset


def load_csv(csv_filename):

    n_component = []
    mean = []
    std = []

    with open(csv_filename, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        csvreader.next()  # we bin the first row of comments
        for row in csvreader:
            n_component.append(int(row[0]))
            mean.append(float(row[1]))
            std.append(float(row[2]))

    return n_component, mean, std


def get_n_component_from_bicfile(bic_filename):
    n_component, mean, std = load_csv(bic_filename)

    return n_component[np.argmin(mean)]


def train_gmm_per_dim(dataset_X, dataset_Y, n_components_per_dim, cv_type='full'):

    # we train one gmm per Y dimension
    gmms = []
    for i in range(dataset_Y.shape[1]):

        X = np.hstack((dataset_X, dataset_Y[:, i][:, np.newaxis]))
        gmm = GMM(n_components=n_components_per_dim[i], covariance_type=cv_type)
        gmm.fit(X)
        gmms.append(gmm)

    return gmms


def train_gmm_full(dataset_X, dataset_Y, n_components=40, cv_type='full'):

    X = np.hstack((dataset_X, dataset_Y))
    gmm = GMM(n_components=n_components, covariance_type=cv_type)
    gmm.fit(X)

    return gmm


def save_model(clfs, filename):
    filetools.ensure_dir(os.path.dirname(filename))
    pickle.dump(clfs, open(filename, "wb"))


if __name__ == '__main__':

    save_folder = os.path.join(HERE_PATH, 'pickled')

    csv_folder = os.path.join(HERE_PATH, 'csv')

    dataset_folder = os.path.join(HERE_PATH, '..', '..', 'datasets')
    datasets = filetools.list_folders(dataset_folder)

    for dataset in datasets:

        dataset_name = os.path.split(dataset)[1]
        pickled_foldername = os.path.join(save_folder, dataset_name)
        filetools.ensure_dir(pickled_foldername)

        (dataset_X, dataset_Y, dataset_info) = load_dataset(dataset_name)

        per_dim_n_component = []
        for i in range(dataset_Y.shape[1]):
            bic_filename = 'gmm_bics_dim_{}.csv'.format(i)
            bic_file = os.path.join(csv_folder, dataset_name, bic_filename)

            n_component = get_n_component_from_bicfile(bic_file)
            per_dim_n_component.append(n_component)

        gmms = train_gmm_per_dim(dataset_X, dataset_Y, per_dim_n_component)
        save_model(gmms, os.path.join(pickled_foldername, 'GMM_per_dim.pkl'))

        #
        bic_filename = 'gmm_bics_dim_full.csv'
        bic_file = os.path.join(csv_folder, dataset_name, bic_filename)
        n_component = get_n_component_from_bicfile(bic_file)

        gmm = train_gmm_full(dataset_X, dataset_Y, n_component)
        save_model(gmm, os.path.join(pickled_foldername, 'GMM_full.pkl'))
