# showcase of the gridea parameter tuning

from deap import benchmarks

from ga.ga import GA
import tools


def problem_function(individual):
    value = benchmarks.bohachevsky(individual)
    return -value[0]


def scoring_function(logbook):
    return logbook[-1]['median']


param_grid = {'pop_size': [20],
              'part_dim': [2],
              'pmin': [-1],
              'pmax': [1],
              'n_survivors': [2, 3, 5],
              'per_locus_rate': [0.1, 0.25, 0.5, 1],
              'per_locus_SD': [0.01, 0.1, 1],
              'temp': [0.1, 0.5]}

gridEA_param = {'optimizor': GA,
                'param_grid': param_grid,
                'problem_function': problem_function,
                'scoring_function': scoring_function,
                'n_generation': 50,
                'n_repeats': 20}

gridEA = tools.GridSearchEA(**gridEA_param)

best_params = gridEA.run()

print best_params
