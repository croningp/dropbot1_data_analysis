import random

import numpy

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


def soft_max(values, temp):
    x = numpy.array(values)
    e = numpy.exp(x / temp)
    soft_max_x = e / numpy.sum(e, axis=0)
    return list(soft_max_x)


def probabilistic_choice(proba_distribution):
    return int(numpy.random.choice(range(len(proba_distribution)), 1, p=proba_distribution))


def proba_normalize(x):
    x = numpy.array(x)
    if numpy.sum(x) == 0:
        x = numpy.ones(x.shape)
    return list(x / numpy.sum(x))


class GA(object):
    """
    A strategy that will keep track of the basic parameters of the GA
    algorithm.
    :param pop_size: Size of the population
    :param part_dim: Number of dimension per genome
    :param pmin: Min value a genome can take for each dimension
    :param pmax: Max value a genome can take for each dimension
    :param n_survivors: Number of survivors to next generation
    :param temp: Temperature for the softmax at survival selection step (default 1)
    """

    def __init__(self, pop_size, part_dim, pmin, pmax, n_survivors, *args, **kargs):

        self.pop_size = pop_size
        self.part_dim = part_dim
        self.pmin = pmin
        self.pmax = pmax
        self.n_survivors = n_survivors

        self.temp = float(kargs.get('temp', 1))

        self.setup()
        self.generate_init_population()

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

    def generate_init_population(self):
        self.population = [self.generate_genome() for _ in range(self.pop_size)]

    def generate_genome(self):
        genome = creator.Individual(random.uniform(self.pmin, self.pmax) for _ in range(self.part_dim))
        return genome

    def get_next_population(self):
        return self.population

    def set_fitness_value(self, fitnesses):
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = (fit, )

        log_population = [list(x) for x in self.population]
        record = self.stats.compile(self.population)
        self.logbook.record(gen=self.gen, nevals=len(self.population), population=log_population, fitnesses=fitnesses, **record)

        self.update_population()
        self.gen += 1

    def get_fitnesses(self):
        fitnesses = []
        for ind in self.population:
            fitnesses.append(ind.fitness.values[0])
        return fitnesses

    def update_population(self):
        survivors_id = self.draw_survivors_id()

        new_population = [self.generate_genome() for _ in range(self.pop_size)]

        for i in survivors_id:
            new_population[i] = self.population[i]

        self.population = new_population

    def draw_survivors_id(self):
        fitnesses = self.get_fitnesses()
        x = soft_max(fitnesses, self.temp)

        survivors_id = []
        for _ in range(self.n_survivors):
            x = proba_normalize(x)
            ind = probabilistic_choice(x)

            survivors_id.append(ind)
            x[ind] = 0

        return survivors_id


if __name__ == "__main__":

    current_problem = benchmarks.bohachevsky

    def my_evalution_function(my_input):
        minput = [my_input[i] for i in [0, 1]]
        value = current_problem(minput)
        return -value[0]

    params = {'pop_size': 20,
              'part_dim': 3,
              'pmin': -1,
              'pmax': 1,
              'n_survivors': 5,
              'temp': 0.1}

    optimizer = GA(**params)

    for _ in range(60):
        pop = optimizer.get_next_population()

        fitnesses = []
        for ind in pop:
            fitnesses.append(my_evalution_function(ind))

        optimizer.set_fitness_value(fitnesses)
