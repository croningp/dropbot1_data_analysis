import operator
import random

import numpy

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


class PSO(object):
    """
    A strategy that will keep track of the basic parameters of the PSO
    algorithm.
    :param pop_size: Size of the population
    :param part_dim: Number of dimension per particle
    :param pmin: Min value a particle can take for each dimension
    :param pmax: Max value a particle can take for each dimension
    :param max_abs_speed: Max aboslute speed of particule (default to 1)
    :param phi1: Max bound for the U vector for speed update, U is sample randomly in [0, phi1], phi1/2 represents the mean stiffness of the springs pulling a particle (default to 2)  attraction drive to topological neighborhood
    :param phi2: Idem as phi1 (default to 2), attraction drive to best point found by any member
    See Particle Swarm Optimization an Overview, Riccardo Poli, James Kennedy, Tim Blackwell, 2007
    :param w_start: inertia weight initial value (default to 1)
    :param w_decay: inertia weight multiplication constant each run (default to 1)
    See Shi, Y., & Eberhart, R. C. (1998). A modified particle swarm optimizer.
    """

    def __init__(self, pop_size, part_dim, pmin, pmax, *args, **kargs):

        self.pop_size = pop_size
        self.part_dim = part_dim
        self.pmin = pmin
        self.pmax = pmax

        self.max_abs_speed = kargs.get('max_abs_speed', 1)
        self.smin = -self.max_abs_speed
        self.smax = self.max_abs_speed
        self.phi1 = kargs.get('phi1', 2)
        self.phi2 = kargs.get('phi1', 2)
        self.w_start = kargs.get('w_start', 1)
        self.w_decay = kargs.get('w_decay', 1)

        self.setup()
        self.generate_init_population()

    def setup(self):
        # We are facing a maximization problem
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # Particle have attribute like speed
        creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, best=None)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", numpy.mean)
        self.stats.register("median", numpy.median)
        self.stats.register("std", numpy.std)
        self.stats.register("min", numpy.min)
        self.stats.register("max", numpy.max)

        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals', 'population', 'fitnesses', 'w'] + self.stats.fields

        self.gen = 0
        self.best = None
        self.w = self.w_start

    def generate_init_population(self):
        self.population = [self.generate_particle() for _ in range(self.pop_size)]

    def generate_particle(self):
        part = creator.Particle(random.uniform(self.pmin, self.pmax) for _ in range(self.part_dim))
        part.speed = [random.uniform(self.smin, self.smax) for _ in range(self.part_dim)]
        return part

    def get_next_population(self):
        return self.population

    def set_fitness_value(self, fitnesses):
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = (fit, )

        log_population = [list(x) for x in self.population]
        record = self.stats.compile(self.population)
        self.logbook.record(gen=self.gen, nevals=len(self.population), population=log_population, fitnesses=fitnesses, w=self.w, **record)

        self.update_population()
        self.update_w()
        self.gen += 1

    def update_w(self):
        self.w = self.w * self.w_decay

    def update_best(self):
        for part in self.population:
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not self.best or self.best.fitness < part.fitness:
                self.best = creator.Particle(part)
                self.best.fitness.values = part.fitness.values

    def update_particle(self, part):
        # sample update weight
        u1 = (random.uniform(0, self.phi1) for _ in range(len(part)))
        u2 = (random.uniform(0, self.phi2) for _ in range(len(part)))
        # compute speed updates
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, self.best, part))
        # apply updates (note self.w is use too here)
        part.speed = list(map(operator.add, map(operator.mul, [self.w] * self.part_dim, part.speed), map(operator.add, v_u1, v_u2)))
        # constrain speed within limits
        for i, speed in enumerate(part.speed):
            part.speed[i] = float(numpy.clip(speed, self.smin, self.smax))
        # apply speed
        part[:] = list(map(operator.add, part, part.speed))
        # constrain value
        for i, value in enumerate(part):
            part[i] = float(numpy.clip(value, self.pmin, self.pmax))
        raw_input()

    def update_population(self):
        self.update_best()
        for part in self.population:
            self.update_particle(part)


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
              'max_abs_speed': 0.1,
              'phi1': 2,
              'phi2': 2,
              'w_start': 1,
              'w_decay': 0.99}

    optimizer = PSO(**params)

    for _ in range(60):
        pop = optimizer.get_next_population()

        fitnesses = []
        for ind in pop:
            fitnesses.append(my_evalution_function(ind))

        optimizer.set_fitness_value(fitnesses)
