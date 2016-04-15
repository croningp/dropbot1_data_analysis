import os
import json
import numpy as np

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


def scoring_function_end_max(logbook):
    return logbook[-1]['max']


def scoring_function_end_median(logbook):
    return logbook[-1]['median']


def scoring_function_integral_median(logbook):
    medians = []
    for i in range(len(logbook)):
        medians.append(logbook[i]['median'])
    return np.sum(medians)


def save_json_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':

    from ga.ga import GA
    from cmaes.cmaes import CMAES
    from pso.pso import PSO

    import tools

    n_dim = 4
    pop_size = 20
    n_generation = 25
    n_repeats = 100

    ga_param_grid = {'pop_size': [pop_size],
                     'part_dim': [n_dim],
                     'pmin': [0],
                     'pmax': [1],
                     'n_survivors': [2, 3, 5],
                     'per_locus_rate': [0.1, 0.2, 0.3, 0.5],
                     'per_locus_SD': [0.01, 0.05, 0.1, 0.5, 1],
                     'temp': [0.1, 0.3, 0.5, 0.7, 1]}

    cmaes_param_grid = {'lambda_': [pop_size],  # population size
                        'centroid': [[0.5] * n_dim],  # initial mean
                        'sigma': [0.01, 0.05, 0.1, 0.5, 1],  # initial cov
                        'mu': [1, 2, 3, 5]}  # number of survivors

    pso_param_grid = {'pop_size': [pop_size],
                      'part_dim': [n_dim],
                      'pmin': [0],
                      'pmax': [1],
                      'max_abs_speed': [0.01, 0.05, 0.1, 0.2],
                      'phi1': [1, 1.5, 2],
                      'phi2': [1, 1.5, 2],
                      'w_start': [0.9, 1, 1.1],
                      'w_decay': [0.5, 0.9, 0.99, 1]}

    method_names = ['GA', 'CMAES', 'PSO']
    optimizators = [GA, CMAES, PSO]
    param_grids = [ga_param_grid, cmaes_param_grid, pso_param_grid]

    scoring_names = ['max', 'median', 'integral']
    scoring_functions = [scoring_function_end_max, scoring_function_end_median, scoring_function_integral_median]

    #
    save_folder = os.path.join(HERE_PATH, 'pickled')
    filetools.ensure_dir(save_folder)

    for i in range(len(method_names)):
        for j in range(len(scoring_names)):

            gridsearch_param = {'optimizor': optimizators[i],
                                'param_grid': param_grids[i],
                                'problem_function': problem_function,
                                'scoring_function': scoring_functions[j],
                                'n_generation': n_generation,
                                'n_repeats': n_repeats}

            gridEA = tools.GridSearchEA(**gridsearch_param)

            best_params, best_score = gridEA.run()

            # save
            filename = method_names[i] + '_' + scoring_names[j] + '.json'
            path = os.path.join(save_folder, filename)

            results = {'params': best_params, 'best_score': best_score}
            save_json_to_file(results, path)
