import numpy as np

from sklearn.grid_search import ParameterGrid


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

        return self.best_params_
