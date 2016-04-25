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


def division_problem(individual):
    # ["division", "directionality", "movement"]
    outdim = 0  # we go for directionality
    x = np.clip(individual, 0, 1)
    x = proba_normalize(x)
    return model.predict(x, outdim)[0, 0]


def directionality_problem(individual):
    # ["division", "directionality", "movement"]
    outdim = 1  # we go for directionality
    x = np.clip(individual, 0, 1)
    x = proba_normalize(x)
    return model.predict(x, outdim)[0, 0]


def movement_problem(individual):
    # ["division", "directionality", "movement"]
    outdim = 2  # we go for movement
    x = np.clip(individual, 0, 1)
    x = proba_normalize(x)
    return model.predict(x, outdim)[0, 0]


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def plot_results(results, method_names, n_repeats):

    fig = plt.figure(figsize=(12, 8))

    for i in range(len(method_names)):
        method_results = results[method_names[i]]
        y = method_results['median']['mean']
        yerr = method_results['median']['std'] / np.sqrt(n_repeats)
        x = range(1, len(y) + 1)
        plt.errorbar(x, y, yerr=yerr, linewidth=2)

    plt.xlim([0, x[-1] + 1])

    plt.xlabel('Generation number')
    plt.ylabel('Fitness value')
    plt.legend(method_names, bbox_to_anchor=(1, 0.35))
    plt.tight_layout()

    return fig


if __name__ == '__main__':

    import sys
    if len(sys.argv) > 1:
        n_repeats = int(sys.argv[1])
    else:
        n_repeats = 100

    from ga.ga import GA
    from cmaes.cmaes import CMAES
    from pso.pso import PSO

    import tools

    n_experiments = 500

    for pop_size in [5, 10, 20]:

        n_generation = n_experiments / pop_size

        method_names = ['GA', 'CMAES', 'PSO']
        optimizators = [GA, CMAES, PSO]

        root_plot_foldername = os.path.join(HERE_PATH, 'plot')
        filetools.ensure_dir(root_plot_foldername)

        param_foldername = os.path.join(HERE_PATH, 'pickled')

        pop_size_foldername = 'pop_{}'.format(pop_size)
        pop_size_folder = os.path.join(param_foldername, pop_size_foldername)

        problem_folders = filetools.list_folders(pop_size_folder)

        problem_names = ['division', 'directionality', 'movement']
        problems = [division_problem, directionality_problem, movement_problem]

        for problem_folder in problem_folders:

            problem_name = os.path.split(problem_folder)[1]
            plot_foldername = os.path.join(root_plot_foldername, pop_size_foldername, problem_name)
            filetools.ensure_dir(plot_foldername)

            problem_ind = problem_names.index(problem_name)
            problem_function = problems[problem_ind]

            results = {}
            for i in range(len(method_names)):

                param_file = os.path.join(problem_folder, '{}_params.json'.format(method_names[i]))
                param_info = load_json(param_file)

                best_param = param_info['params']

                results[method_names[i]] = tools.run_multiple_ea_and_analyse(optimizators[i], best_param, problem_function, n_generation, n_repeats)

            fig = plot_results(results, method_names, n_repeats)

            # save
            plot_filename = os.path.join(plot_foldername, 'optimized_params.png')

            plt.savefig(plot_filename, dpi=100)
            plt.close()
