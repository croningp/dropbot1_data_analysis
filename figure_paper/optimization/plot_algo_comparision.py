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

    n_repeats = 100
    pop_size = 20
    n_experiments = 500
    n_generation = n_experiments / pop_size

    param_foldername = os.path.join(root_path, 'optimization', 'pickled')
    pop_size_foldername = 'pop_{}'.format(pop_size)
    pop_size_folder = os.path.join(param_foldername, pop_size_foldername)

    problem_folder = os.path.join(pop_size_folder, 'directionality')
    problem_function = directionality_problem

    json_foldername = os.path.join(root_path, 'optimization', 'json')
    brute_force_file = os.path.join(json_foldername, 'brute_force_max.json')
    brute_force_data = load_json(brute_force_file)
    brute_force_max_directionality = brute_force_data['directionality'][1]

    # run and store results
    import optimization.tools
    results = {}

    # optimized GA
    from optimization.ga.ga import GA
    ga_param_file = os.path.join(problem_folder, 'GA_params.json')
    ga_param_info = load_json(ga_param_file)
    ga_param_optimized = ga_param_info['params']
    results['GA'] = optimization.tools.run_multiple_ea_and_analyse(GA, ga_param_optimized, problem_function, n_generation, n_repeats)

    # optimized PSO
    from optimization.pso.pso import PSO
    pso_param_file = os.path.join(problem_folder, 'PSO_params.json')
    pso_param_info = load_json(pso_param_file)
    pso_param_optimized = pso_param_info['params']
    results['PSO'] = optimization.tools.run_multiple_ea_and_analyse(PSO, pso_param_optimized, problem_function, n_generation, n_repeats)

    # optimized CMAES
    from optimization.cmaes.cmaes import CMAES
    cmaes_param_file = os.path.join(problem_folder, 'CMAES_params.json')
    cmaes_param_info = load_json(cmaes_param_file)
    cmaes_param_optimized = cmaes_param_info['params']
    results['CMAES'] = optimization.tools.run_multiple_ea_and_analyse(CMAES, cmaes_param_optimized, problem_function, n_generation, n_repeats)

    # design figure
    fontsize = 22
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    matplotlib.rcParams.update({'font.size': fontsize})

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    method_names = ['GA', 'PSO', 'CMAES']
    for i in range(len(results)):
        method_results = results[method_names[i]]
        y = method_results['median']['mean']
        yerr = method_results['median']['std'] / np.sqrt(n_repeats)
        x = range(1, len(y) + 1)
        plt.errorbar(x, y, yerr=yerr, linewidth=2)

    plt.plot([0, x[-1] + 1], [brute_force_max_directionality, brute_force_max_directionality], color='grey', linestyle='--')
    text = 'Max achievable: ' + str(round(brute_force_max_directionality, 2))
    ax.text(0.1, 7.9, text, color='grey', fontsize=15)

    plt.xlim([0, x[-1] + 1])
    plt.ylim([4, 8.5])

    plt.xlabel('Generation number', fontsize=fontsize)
    plt.ylabel('Fitness value', fontsize=fontsize)
    plt.legend(method_names, bbox_to_anchor=(1, 0.25), fontsize=fontsize)
    plt.tight_layout()

    # save
    plot_foldername = os.path.join(HERE_PATH, 'plot')
    filetools.ensure_dir(plot_foldername)

    for ext in ['.png', '.eps', '.svg']:
        plot_filename = 'algo_comparison' + ext

        filepath = os.path.join(plot_foldername, plot_filename)
        plt.savefig(filepath, dpi=100)

    plt.close()
