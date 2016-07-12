import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..')
sys.path.append(root_path)

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from utils import filetools

# design figure
fontsize = 22
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rcParams.update({'font.size': fontsize})


def soft_max(values, temp):
    x = np.array(values)
    e = np.exp(x / temp)
    soft_max_x = e / np.sum(e, axis=0)
    return list(soft_max_x)


def proba_normalize(x):
    x = np.array(x, dtype=float)
    if np.sum(x) == 0:
        x = np.ones(x.shape)
    return x / np.sum(x, dtype=float)


def save_and_close_figure(filebasename, exts=['.png'], dpi=100):

    for ext in exts:
        filepath = filebasename + ext
        plt.savefig(filepath, dpi=dpi)
    plt.close()


fitnesses = {}
fitnesses['not_much_problem'] = [8, 5, 2, 1.8, 1.5, 1, 0.5, 0.3, 0.3, 0.2]
fitnesses['problem'] = [8 + i for i in fitnesses['not_much_problem']]
fitnesses['random'] = list(10 * np.sort(np.random.rand(20))[-1:0:-1])


for k, fitness in fitnesses.items():
    savefolder = os.path.join(HERE_PATH, k)
    filetools.ensure_dir(savefolder)

    roulette = proba_normalize(fitness)

    softmaxlist = []
    temperaturelist = np.logspace(-2, 1, 22)
    for t in temperaturelist:
        softmaxlist.append(soft_max(fitness, t))

    x = range(1, len(fitness) + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(x, fitness)
    plt.xticks(x, x)
    plt.xlabel('Individual Number', fontsize=fontsize)
    plt.ylabel('Fitness Value', fontsize=fontsize)
    plt.xlim([0.9, len(x) + 0.1])
    plt.ylim([0, 21])
    save_and_close_figure(os.path.join(savefolder, 'fitness'))

    plt.figure(figsize=(12, 8))
    plt.plot(x, roulette)
    plt.xticks(x, x)
    plt.xlabel('Individual Number', fontsize=fontsize)
    plt.ylabel('Probability', fontsize=fontsize)
    plt.title('Fitness normalized', fontsize=fontsize)
    plt.xlim([0.9, len(x) + 0.1])
    plt.ylim([-0.1, 1.1])
    save_and_close_figure(os.path.join(savefolder, 'normalized'))

    for i in range(len(temperaturelist)):

        t = temperaturelist[i]
        softvalues = softmaxlist[i]

        plt.figure(figsize=(12, 8))
        plt.plot(x, softvalues)
        plt.xticks(x, x)
        plt.xlabel('Individual Number', fontsize=fontsize)
        plt.ylabel('Probability', fontsize=fontsize)
        plt.title('Fitness softmax with T={}'.format(t), fontsize=fontsize)
        plt.xlim([0.9, len(x) + 0.1])
        plt.ylim([-0.1, 1.1])
        save_and_close_figure(os.path.join(savefolder, 'softmax_t={}'.format(t)))
