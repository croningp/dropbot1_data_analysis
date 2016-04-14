import numpy
import random

import pickle

from deap import base
from deap import cma
from deap import creator
from deap import tools


class CMAES(cma.Strategy):

    def __init__(self, *args, **kargs):
        super(CMAES, self).__init__(*args, **kargs)
        self.setup()

    def setup(self):
        # We are facing a maximization problem
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", numpy.mean)
        self.stats.register("median", numpy.median)
        self.stats.register("std", numpy.std)
        self.stats.register("min", numpy.min)
        self.stats.register("max", numpy.max)

        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals', 'population', 'fitnesses'] + self.stats.fields

        self.gen = 0

    def get_next_population(self):
        self.population = self.generate(creator.Individual)
        return self.population

    def set_fitness_value(self, fitnesses):
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = (fit, )

        self.update(self.population)

        log_population = [list(x) for x in self.population]
        record = self.stats.compile(self.population)
        self.logbook.record(gen=self.gen, nevals=len(self.population), population=log_population, fitnesses=fitnesses, **record)

        self.gen += 1

    def forge_checkpoint(self):
        checkpoint = {}

        checkpoint['rnd_state'] = random.getstate()
        checkpoint['numpy_rnd_state'] = numpy.random.get_state()

        attribute_to_checkpoint = [
            'logbook',
            'gen',
            'params',
            'centroid',
            'dim',
            'sigma',
            'pc',
            'ps',
            'chiN',
            'C',
            'diagD',
            'B',
            'BD',
            'cond',
            'lambda_',
            'update_count',
            'mu',
            'weights',
            'mueff',
            'cc',
            'cs',
            'ccov1',
            'ccovmu',
            'damps']

        for key in attribute_to_checkpoint:
            checkpoint[key] = getattr(self, key)

        return checkpoint

    def apply_checkpoint(self, checkpoint):
        for key, value in checkpoint.items():
            if key == 'rnd_state':
                random.setstate(value)
            elif key == 'numpy_rnd_state':
                numpy.random.set_state(value)
            else:
                setattr(self, key, value)

    def save_checkpoint_to_file(self, filename):
        checkpoint = self.forge_checkpoint()
        pickle.dump(checkpoint, open(filename, 'w'))

    @classmethod
    def from_checkpoint(cls, checkpoint):
        optimizer = cls(checkpoint['centroid'], checkpoint['sigma'])
        optimizer.apply_checkpoint(checkpoint)
        return optimizer

    @classmethod
    def from_checkpoint_file(cls, filename):
        checkpoint = pickle.load(open(filename))
        return cls.from_checkpoint(checkpoint)

if __name__ == '__main__':

    from deap import benchmarks

    search_space_dims = 2

    current_problem = benchmarks.bohachevsky

    def my_evalution_function(my_input):
        value = current_problem(my_input)
        return -value[0]

    params = {'centroid': [5.0] * search_space_dims,
              'sigma': 10,
              'lambda_': 5 * search_space_dims,
              'mu': 1}

    optimizer = CMAES(**params)

    for i in xrange(60):
        pop = optimizer.get_next_population()

        fitnesses = []
        for ind in pop:
            fitnesses.append(my_evalution_function(ind))

        optimizer.set_fitness_value(fitnesses)

    print optimizer.logbook[-1]['fitnesses']
