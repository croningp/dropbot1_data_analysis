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


def scoring_function(logbook):
    end_median = scoring_function_end_median(logbook)
    mean_median = scoring_function_mean_median(logbook)
    return end_median + mean_median


def scoring_function_end_max(logbook):
    return logbook[-1]['max']


def scoring_function_end_median(logbook):
    return logbook[-1]['median']


def scoring_function_mean_median(logbook):
    medians = []
    for i in range(len(logbook)):
        medians.append(logbook[i]['median'])
    return np.mean(medians)


def save_json_to_file(data, filename):
    filetools.ensure_dir(os.path.dirname(filename))
    with open(filename, 'w') as f:
        json.dump(data, f)

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

    save_folder = os.path.join(HERE_PATH, 'pickled')
    filetools.ensure_dir(save_folder)

    n_dim = 4
    n_experiments = 500

    for pop_size in [5, 10, 20]:

        n_generation = n_experiments / pop_size

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
                            'weights': ['superlinear', 'linear', 'equal']}  # decrease speed

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

        problem_names = ['division', 'directionality', 'movement']
        problems = [division_problem, directionality_problem, movement_problem]

        for i in range(len(method_names)):
            for j in range(len(problem_names)):

                pop_size_str = 'pop_{}'.format(pop_size)
                filename = method_names[i] + '_params.json'
                save_path = os.path.join(save_folder, pop_size_str, problem_names[j], filename)

                if not os.path.exists(save_path):
                    gridsearch_param = {'optimizor': optimizators[i],
                                        'param_grid': param_grids[i],
                                        'problem_function': problems[j],
                                        'scoring_function': scoring_function,
                                        'n_generation': n_generation,
                                        'n_repeats': n_repeats}

                    gridEA = tools.GridSearchEA(**gridsearch_param)

                    best_params, best_score = gridEA.run()

                    # save
                    results = {'params': best_params, 'best_score': best_score, 'grid_scores': gridEA.grid_scores_}
                    save_json_to_file(results, save_path)

                else:
                    print '###\n[{}, {}, {}] already done, please delete this file to run again: {}'.format(pop_size_str, method_names[i], problem_names[j], save_path)
