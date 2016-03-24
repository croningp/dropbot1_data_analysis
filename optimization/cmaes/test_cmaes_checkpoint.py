import cmaes
import random
import numpy

from deap import benchmarks

search_space_dims = 2
n_run = 60

current_problem = benchmarks.bohachevsky


def my_evalution_function(my_input):
    value = current_problem(my_input)
    return -value[0]


# full run without checkpoint
random.seed(0)
numpy.random.seed(0)
optimizer = cmaes.CMAES(centroid=[5.0] * search_space_dims,
                        sigma=10,
                        lambda_=5 * search_space_dims,
                        mu=5)

for i in xrange(n_run):
    pop = optimizer.get_next_population()

    fitnesses = []
    for ind in pop:
        fitnesses.append(my_evalution_function(ind))

    optimizer.set_fitness_value(fitnesses)

print 'End results without checkpoint:'
print 'Final population is ', optimizer.logbook[-1]['population']
print 'Final fitnesses is ', optimizer.logbook[-1]['fitnesses']
print ''

no_checkpoint_logbook = optimizer.logbook

# same run but saving and loading from check point each time
checkpoint_filename = 'test_cmaes_pickle_checkpoint.pkl'

random.seed(0)
numpy.random.seed(0)
optimizer = cmaes.CMAES(centroid=[5.0] * search_space_dims,
                        sigma=10,
                        lambda_=5 * search_space_dims,
                        mu=5)

for i in xrange(n_run):
    pop = optimizer.get_next_population()

    fitnesses = []
    for ind in pop:
        fitnesses.append(my_evalution_function(ind))

    optimizer.set_fitness_value(fitnesses)

    checkpoint = optimizer.forge_checkpoint()

    # save the current state
    import pickle
    pickle.dump(checkpoint, open(checkpoint_filename, 'w'))

    # play with the random generator so we ensure our load checkpoint works fine
    random.random()
    numpy.random.rand()
    del(optimizer)

    # reload the state from the file
    optimizer = cmaes.CMAES.from_checkpoint_file(checkpoint_filename)

print 'End results with checkpoint:'
print 'Final population is ', optimizer.logbook[-1]['population']
print 'Final fitnesses is ', optimizer.logbook[-1]['fitnesses']
print ''

#
checkpoint_logbook = optimizer.logbook
if checkpoint_logbook == no_checkpoint_logbook:
    print 'Both result seems to be exactly the same!'
else:
    print 'Hmmm the experiment where not repeated exactly..'
