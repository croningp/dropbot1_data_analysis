import numpy as np

from sklearn.grid_search import ParameterGrid


def run_multiple_ea_and_analyse(optimizator, param, problem_function, n_generation, n_repeats, fields=['avg', 'median', 'max', 'min']):

    repeat_results = {}
    for k in fields:
        repeat_results[k] = []

    for i in range(n_repeats):

        logbook = run_ea(optimizator, param, problem_function, n_generation)

        analysis_info = analyse_logbook(logbook)

        for k in fields:
            repeat_results[k].append(analysis_info[k])

    collapsed_results = {}
    for k in fields:
        collapsed_results[k] = {}
        collapsed_results[k]['mean'] = np.mean(repeat_results[k], axis=0)
        collapsed_results[k]['std'] = np.std(repeat_results[k], axis=0)

    return collapsed_results


def run_ea(optimizator, param, problem_function, n_generation):

    optimizer = optimizator(**param)

    for _ in range(n_generation):
        population = optimizer.get_next_population()

        fitnesses = []
        for individual in population:
            fitnesses.append(problem_function(individual))
        optimizer.set_fitness_value(fitnesses)

    return optimizer.logbook


def analyse_logbook(logbook, fields=['avg', 'median', 'max', 'min']):

    results = {}
    for k in fields:
        results[k] = []

    for i, log in enumerate(logbook):
        for k in fields:
            results[k].append(log[k])

    return results


def run_multiple_ea_and_concatenate_fitnesses(optimizator, param, problem_function, n_generation, n_repeats):

    concatenated_fitnesses = []

    for i in range(n_repeats):

        logbook = run_ea(optimizator, param, problem_function, n_generation)

        fitnesses = concatenate_fitnesses(logbook)

        concatenated_fitnesses.append(fitnesses)

    return concatenated_fitnesses


def concatenate_fitnesses(logbook):

    fitnesses = []
    for i, log in enumerate(logbook):
        for fitness in log['fitnesses']:
            fitnesses.append(fitness)
    return fitnesses


class GridSearchEA(object):

    def __init__(self, optimizor, param_grid, problem_function, scoring_function, n_generation, n_repeats=100):

        self.optimizor = optimizor

        self.param_grid = param_grid
        self.param_product = list(ParameterGrid(param_grid))

        self.problem_function = problem_function
        self.scoring_function = scoring_function

        self.n_generation = n_generation
        self.n_repeats = n_repeats

    def run_one_param_config(self, param_config):

        optimizer = self.optimizor(**param_config)

        for _ in range(self.n_generation):
            population = optimizer.get_next_population()

            fitnesses = []
            for individual in population:
                fitnesses.append(self.problem_function(individual))
            optimizer.set_fitness_value(fitnesses)

        return self.scoring_function(optimizer.logbook)

    def run_repeats_param_config(self, param_config):

        scores = []
        for _ in range(self.n_repeats):
            scores.append(self.run_one_param_config(param_config))
        return scores

    def run(self, verbose=True):

        self.grid_scores_ = []
        self.best_score_ = None
        self.best_params_ = None

        for i, param_config in enumerate(self.param_product):

            if verbose:
                print '\n###\nRunning {}/{}'.format(i + 1, len(self.param_product))
                print 'Params: {}'.format(param_config)

            scores = self.run_repeats_param_config(param_config)

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            if verbose:
                print 'Score: Mean={} STD={}'.format(mean_score, std_score)

            result = {'mean': mean_score,
                      'std': std_score,
                      'params': param_config}

            self.grid_scores_.append(result)

            if self.best_score_ is None or self.best_score_ < mean_score:
                self.best_score_ = mean_score
                self.best_params_ = param_config

        return self.best_params_, self.best_score_
