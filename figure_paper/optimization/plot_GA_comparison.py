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

    import random
    random.seed(0)
    np.random.seed(0)

    from optimization.ga.ga import GA
    from optimization.ga.ga_dropbot1 import GA_DROPBOT1

    import optimization.tools

    n_repeats = 100
    pop_size = 20
    n_experiments = 500
    n_generation = n_experiments / pop_size

    param_foldername = os.path.join(root_path, 'optimization', 'pickled')
    pop_size_foldername = 'pop_{}'.format(pop_size)
    pop_size_folder = os.path.join(param_foldername, pop_size_foldername)

    problem_folder = os.path.join(pop_size_folder, 'directionality')
    problem_function = directionality_problem

    # optimized GA param
    ga_param_file = os.path.join(problem_folder, 'GA_params.json')
    ga_param_info = load_json(ga_param_file)
    ga_param_optimized = ga_param_info['params']

    # the parameters below are the one the most similar to the intial setup
    ga_param_dropbot1 = {'pop_size': 20,
                         'part_dim': 4,
                         'pmin': 0,
                         'pmax': 1,
                         'n_survivors': 12,  # was 15 out of 25, so 12 out of 20
                         'per_locus_rate': 0.3,  # exactly the same
                         'per_locus_SD': 0.1}  # exactly the same

    ga_param_dropbot1_softmax = {'pop_size': 20,
                                 'part_dim': 4,
                                 'pmin': 0,
                                 'pmax': 1,
                                 'n_survivors': 12,  # was 15 out of 25, so 12 out of 20
                                 'per_locus_rate': 0.3,  # exactly the same
                                 'per_locus_SD': 0.1,  # exactly the same
                                 'temp': 1}  # default temperature value for softmax

    # run and store results
    results = {}
    results['GA_optimized'] = optimization.tools.run_multiple_ea_and_analyse(GA, ga_param_optimized, problem_function, n_generation, n_repeats)
    results['GA_origin_softmax'] = optimization.tools.run_multiple_ea_and_analyse(GA, ga_param_dropbot1, problem_function, n_generation, n_repeats)
    results['GA_origin'] = optimization.tools.run_multiple_ea_and_analyse(GA_DROPBOT1, ga_param_dropbot1, problem_function, n_generation, n_repeats)

    # design figure
    fontsize = 22
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    matplotlib.rcParams.update({'font.size': fontsize})

    fig = plt.figure(figsize=(12, 8))

    method_names = ['GA_optimized', 'GA_origin_softmax', 'GA_origin']
    for i in range(len(results)):
        method_results = results[method_names[i]]
        y = method_results['median']['mean']
        yerr = method_results['median']['std'] / np.sqrt(n_repeats)
        x = range(1, len(y) + 1)
        plt.errorbar(x, y, yerr=yerr, linewidth=2)

    plt.xlim([0, x[-1] + 1])

    plt.xlabel('Generation number', fontsize=fontsize)
    plt.ylabel('Fitness value', fontsize=fontsize)
    plt.legend(method_names, bbox_to_anchor=(1, 0.25), fontsize=fontsize)
    plt.tight_layout()

    # save
    plot_foldername = os.path.join(HERE_PATH, 'plot')
    filetools.ensure_dir(plot_foldername)

    for ext in ['.png', '.eps', '.svg']:
        plot_filename = 'ga_comparison' + ext

        filepath = os.path.join(plot_foldername, plot_filename)
        plt.savefig(filepath, dpi=100)

    plt.close()
