import os
import json
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..', '..')
sys.path.append(root_path)

from utils import filetools

import models.regression.tools
kernelridgerbf_file = os.path.join(root_path, 'models', 'regression', 'pickled', 'octanoic', 'KernelRidge-RBF.pkl')
model = models.regression.tools.load_model(kernelridgerbf_file)


def proba_normalize(x):
    x = np.array(x, dtype=float)
    if np.sum(x) == 0:
        x = np.ones(x.shape)
    return x / np.sum(x, dtype=float)


def directionality_problem(individual):
    # ["division", "directionality", "movement"]
    outdim = 1  # we go for directionality
    x = np.clip(individual, 0, 1)
    x = proba_normalize(x)
    return model.predict(x, outdim)[0, 0]


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


if __name__ == '__main__':

    n_repeats = 100

    from optimization.cmaes.cmaes import CMAES
    import optimization.tools

    n_experiments = 500

    ##
    method_name = 'CMAES'
    optimizator = CMAES

    problem_name = 'directionality'
    problem_function = directionality_problem

    json_foldername = os.path.join(root_path, 'optimization', 'json')
    brute_force_file = os.path.join(json_foldername, 'brute_force_max.json')
    brute_force_data = load_json(brute_force_file)
    brute_force_max_directionality = brute_force_data['directionality'][1]

    pop_sizes = [5, 10, 20]

    results = []
    for pop_size in pop_sizes:

        n_generation = n_experiments / pop_size

        ##
        param_foldername = os.path.join(root_path, 'optimization', 'pickled')

        pop_size_foldername = 'pop_{}'.format(pop_size)
        pop_size_folder = os.path.join(param_foldername, pop_size_foldername)

        ##
        param_file = os.path.join(pop_size_folder, problem_name, '{}_params.json'.format(method_name))
        param_info = load_json(param_file)

        best_param = param_info['params']

        results.append(optimization.tools.run_multiple_ea_and_concatenate_fitnesses(optimizator, best_param, problem_function, n_generation, n_repeats))

    # design figure
    fontsize = 22
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    matplotlib.rcParams.update({'font.size': fontsize})

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    for i in range(len(results)):
        data = np.array(results[i])
        x = range(1, data.shape[1] + 1)
        y = data.mean(axis=0)
        plt.plot(x, y, linewidth=2)
        # yerr = data.std(axis=0) / np.sqrt(n_repeats)
        # plt.errorbar(x, y, yerr=yerr, linewidth=2, errorevery=20)

    plt.plot([0, x[-1] + 1], [brute_force_max_directionality, brute_force_max_directionality], color='grey', linestyle='--')
    text = 'Max achievable: ' + str(round(brute_force_max_directionality, 2))
    ax.text(2, 7.9, text, color='grey', fontsize=15)

    plt.xlim([0, x[-1] + 1])
    plt.ylim([4, 8.5])

    plt.xlabel('Experiment number', fontsize=fontsize)
    plt.ylabel('Fitness value', fontsize=fontsize)
    legend_name = ['pop_{}'.format(p) for p in pop_sizes]
    plt.legend(legend_name, bbox_to_anchor=(1, 0.35), fontsize=fontsize)
    plt.tight_layout()

    # save
    plot_foldername = os.path.join(HERE_PATH, 'plot')
    filetools.ensure_dir(plot_foldername)

    for ext in ['.png', '.eps', '.svg']:
        plot_filename = 'cmaes_pop_comparison' + ext

        filepath = os.path.join(plot_foldername, plot_filename)
        plt.savefig(filepath, dpi=100)

    plt.close()
