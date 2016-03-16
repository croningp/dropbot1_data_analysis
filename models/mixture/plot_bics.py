import os
import csv
import numpy as np

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..', '..')
sys.path.append(root_path)

from utils import filetools

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

matplotlib.rcParams.update({'font.size': 22})


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


def plot_bics(n_component, mean, std):
    plt.figure(figsize=(20, 8))
    ax = plt.subplot(1, 1, 1)

    ax.errorbar(n_component, mean, std, linewidth=2)

    n_comp_min = n_component[np.argmin(mean)]
    ticks = [0, n_comp_min, n_component[-1]]
    plt.xticks(ticks)
    plt.xlim((0, n_component[-1]))

    ax.plot([n_comp_min, n_comp_min], ax.get_ylim(), 'r', linewidth=2)

    plt.xlabel('Number of Gaussians in GMM')
    plt.ylabel('Bic score')


if __name__ == '__main__':

    csv_folder = os.path.join(HERE_PATH, 'csv')
    plot_bic_folder = os.path.join(HERE_PATH, 'plot', 'bics')

    datasets = filetools.list_folders(csv_folder)

    for dataset in datasets:

        dataset_name = os.path.split(dataset)[1]
        plot_foldername = os.path.join(plot_bic_folder, dataset_name)
        filetools.ensure_dir(plot_foldername)

        csv_files = filetools.list_files(dataset)

        for csv_file in csv_files:
            n_component, mean, std = load_csv(csv_file)

            plot_bics(n_component, mean, std)

            # save
            csv_filename = os.path.basename(csv_file)
            (fname, ext) = os.path.splitext(csv_filename)
            bic_plot_filename = os.path.join(plot_foldername, fname + '.png')

            plt.savefig(bic_plot_filename, dpi=100)
            plt.close()
