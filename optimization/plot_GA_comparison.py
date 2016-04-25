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

    import tools

    n_experiments = 500
    pop_size = 20
    n_generation = n_experiments / pop_size

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
        plot_foldername = os.path.join(root_plot_foldername)
        filetools.ensure_dir(plot_foldername)

        problem_ind = problem_names.index(problem_name)
        problem_function = problems[problem_ind]

        ##
        from ga.ga_dropbot1 import GA_DROPBOT1

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

        results = {}
        results['GA_optimized'] = tools.run_multiple_ea_and_analyse(GA, ga_param_optimized, problem_function, n_generation, n_repeats)
        results['GA_origin_softmax'] = tools.run_multiple_ea_and_analyse(GA, ga_param_dropbot1, problem_function, n_generation, n_repeats)
        results['GA_origin'] = tools.run_multiple_ea_and_analyse(GA_DROPBOT1, ga_param_dropbot1, problem_function, n_generation, n_repeats)

        fig = plot_results(results, ['GA_optimized', 'GA_origin_softmax', 'GA_origin'], n_repeats)

        plot_filename = os.path.join(plot_foldername, 'ga_comparison_{}.png'.format(problem_name))

        plt.savefig(plot_filename, dpi=100)
        plt.close()
