import os
import csv
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


def compare_gmm_model(X, n_components_range, cv_type='full'):

    bic = []
    for n_components in n_components_range:
        print 'n_components = {}'.format(n_components)
        # Fit a mixture of Gaussians with EM
        gmm = GMM(n_components=n_components, covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))

    return bic


def compare_and_save(X, n_components_range, filename, cv_type='full', n_repeats=20):

    dirname = os.path.dirname(filename)
    filetools.ensure_dir(dirname)

    bics = []
    for j in range(n_repeats):
        print 'Run {}/{}'.format(j + 1, n_repeats)
        bic = compare_gmm_model(X, n_components_range)
        bics.append(bic)

    with open(filename, 'wb') as csvfile:

        csvwriter = csv.writer(csvfile, delimiter=',')
        header = ['n', 'bic_mean', 'bic_std']
        csvwriter.writerow(header)

        bic_mean = np.array(bics).mean(axis=0)
        bic_std = np.array(bics).std(axis=0)

        for k in range(len(n_components_range)):
            data_list = [n_components_range[k], bic_mean[k], bic_std[k]]
            csvwriter.writerow(data_list)


if __name__ == '__main__':

    save_folder = os.path.join(HERE_PATH, 'csv')

    dataset_folder = os.path.join(HERE_PATH, '..', '..', 'datasets')
    datasets = filetools.list_folders(dataset_folder)

    for dataset in datasets:

        dataset_name = os.path.split(dataset)[1]
        csv_foldername = os.path.join(save_folder, dataset_name)
        filetools.ensure_dir(csv_foldername)

        (dataset_X, dataset_Y, dataset_info) = load_dataset(dataset_name)
        n_components_range = range(1, 101)

        # full model
        X = np.hstack((dataset_X, dataset_Y))
        outfile = os.path.join(csv_foldername, 'gmm_bics_dim_full.csv')

        compare_and_save(X, n_components_range, outfile)

        # dim per dim model
        for i in range(dataset_Y.shape[1]):

            X = np.hstack((dataset_X, dataset_Y[:, i][:, np.newaxis]))
            outfile = os.path.join(csv_foldername, 'gmm_bics_dim_{}.csv'.format(i))

            compare_and_save(X, n_components_range, outfile)
