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
    return - model.predict(x, outdim)[0, 0]


def directionality_problem(individual):
    # ["division", "directionality", "movement"]
    outdim = 1  # we go for directionality
    x = np.clip(individual, 0, 1)
    x = proba_normalize(x)
    return - model.predict(x, outdim)[0, 0]


def movement_problem(individual):
    # ["division", "directionality", "movement"]
    outdim = 2  # we go for movement
    x = np.clip(individual, 0, 1)
    x = proba_normalize(x)
    return - model.predict(x, outdim)[0, 0]


def save_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':

    from scipy.optimize import brute
    # brute tries to minimize a function by brute force method
    # we try to maximize, so the function above are now returning minus f(individual)

    x0_directionality = brute(directionality_problem, [(0, 1), (0, 1), (0, 1), (0, 1)])
    best_directionality = -directionality_problem(x0_directionality)

    x0_division = brute(division_problem, [(0, 1), (0, 1), (0, 1), (0, 1)])
    best_division = -division_problem(x0_division)

    x0_movement = brute(movement_problem, [(0, 1), (0, 1), (0, 1), (0, 1)])
    best_movement = -movement_problem(x0_movement)

    results = {}
    results['directionality'] = (list(proba_normalize(x0_directionality)), best_directionality)
    results['division'] = (list(proba_normalize(x0_division)), best_division)
    results['movement'] = (list(proba_normalize(x0_movement)), best_movement)

    # save
    save_path = os.path.join(HERE_PATH, 'json')
    filetools.ensure_dir(save_path)

    filename = 'brute_force_max.json'
    filepath = os.path.join(save_path, filename)

    save_to_file(results, filepath)
