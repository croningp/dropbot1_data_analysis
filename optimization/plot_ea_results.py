import os
import json
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rcParams.update({'font.size': 22})

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..')
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


def problem_function(individual):
    # ["division", "directionality", "movement"]
    outdim = 1  # we go for directionality
    x = np.clip(individual, 0, 1)
    x = proba_normalize(x)
    return model.predict(x, outdim)[0, 0]


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def plot_results(results, method_names):

    fig = plt.figure(figsize=(12, 8))

    for i in range(len(method_names)):
        method_results = results[method_names[i]]
        y = method_results['median']['mean']
        yerr = method_results['median']['std']
        x = range(1, len(y) + 1)
        plt.errorbar(x, y, yerr=yerr, linewidth=2)

    plt.xlim([0, x[-1] + 1])

    plt.xlabel('Generation number')
    plt.ylabel('Fitness value')
    plt.legend(method_names, bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    return fig


if __name__ == '__main__':

    from ga.ga import GA
    from cmaes.cmaes import CMAES
    from pso.pso import PSO

    import tools

    n_generation = 25
    n_repeats = 100

    method_names = ['GA', 'CMAES', 'PSO']
    optimizators = [GA, CMAES, PSO]

    plot_foldername = os.path.join(HERE_PATH, 'plot')
    filetools.ensure_dir(plot_foldername)

    results = {}
    for i in range(len(method_names)):

        param_file = os.path.join(HERE_PATH, 'pickled', '{}_params.json'.format(method_names[i]))
        param_info = load_json(param_file)

        best_param = param_info['params']

        results[method_names[i]] = tools.run_multiple_ea_and_analyse(optimizators[i], best_param, problem_function, n_generation, n_repeats)

    fig = plot_results(results, method_names)

    # save
    plot_filename = os.path.join(plot_foldername, 'optimized_params.png')

    plt.savefig(plot_filename, dpi=100)
    plt.close()

    ##
    ga_param_file = os.path.join(HERE_PATH, 'pickled', 'GA_params.json')
    ga_param_info = load_json(ga_param_file)
    ga_param_optimized = ga_param_info['params']

    # note this is not exactly the GA used for the dropbot1 paper
    # we have an improved version but the parameters below are the one the most similar to the intial setup
    ga_param_dropbot1 = {'pop_size': 20,
                         'part_dim': 4,
                         'pmin': 0,
                         'pmax': 1,
                         'n_survivors': 12,  # was 15 out of 25, so 12 out of 20
                         'per_locus_rate': 0.3,  # exactly the same
                         'per_locus_SD': 0.1,  # exactly the same
                         'temp': 1}  # was not a softmax but a roulette algortihm, so was relatively smooth/flat selection

    results = {}
    results['GA_opt'] = tools.run_multiple_ea_and_analyse(GA, ga_param_optimized, problem_function, n_generation, n_repeats)
    results['GA_origin'] = tools.run_multiple_ea_and_analyse(GA, ga_param_dropbot1, problem_function, n_generation, n_repeats)

    fig = plot_results(results, ['GA_opt', 'GA_origin'])

    plot_filename = os.path.join(plot_foldername, 'ga_comparison.png')

    plt.savefig(plot_filename, dpi=100)
    plt.close()
