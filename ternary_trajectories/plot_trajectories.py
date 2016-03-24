import os
import numpy as np

import matplotlib.pyplot as plt
import ternary

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..')
sys.path.append(root_path)


def sample_along_line(v, dims=[0, 1], n=11):
    """
    worst function ever, sorry
    sample along a line in ternary plot
    v can be any vector, e.g. [0, 0, 0.8, 0]
    dims can only be two indexes
    the function returns n vectors v eqully spaced by changing the two dims to value maitaining sum of v to one, e.g, if dims = [0,1] and n=3 the function returns [[0.2,0,0.8,0],[0.1,0.1,0.8,0],[0,0.2,0.8,0]]
    really unclear!
    """

    kept_index = range(len(v))
    kept_index.remove(dims[0])
    kept_index.remove(dims[1])

    missing_value_to_sum_one = 1.0 - np.array(v)[kept_index].sum()

    space_up = np.linspace(0, missing_value_to_sum_one, n)

    out = []
    for i, value in enumerate(space_up):
        new_v = list(v)
        new_v[dims[0]] = value
        new_v[dims[1]] = missing_value_to_sum_one - value
        out.append(new_v)

    # yeah really bad function
    return np.array(out)


def extract_info_for_plot(points, keys, plot_dims):
    points_for_plot = points[:, plot_dims]
    axis_labels_for_plot = [keys[i] for i in plot_dims]
    return points_for_plot, axis_labels_for_plot


def scatter_ternary_plot(points, axis_labels):

        fontsize = 22

        fig, tax = ternary.figure(scale=1.0)

        tax.scatter(points, marker='D', color='r')

        # make the plot look good
        tax.boundary()
        tax.gridlines(multiple=0.1, color="black")

        # Set Axis labels and Title
        tax.right_axis_label(axis_labels[0], fontsize=fontsize)
        tax.left_axis_label(axis_labels[1], fontsize=fontsize)
        tax.bottom_axis_label(axis_labels[2], fontsize=fontsize)

        ticks = list(np.linspace(0, 1, 11))
        loc = list(np.linspace(0, 1, 11))
        tax.ticks(ticks=ticks, locations=loc, axis='lbr', linewidth=1, multiple=5)

        # Remove default Matplotlib Axes
        tax.clear_matplotlib_ticks()
        fig.patch.set_visible(False)

        return fig, tax


def save_and_close_current_ternary(fig, tax, filename):

    # weird stuff to make it save the plot properly
    # otherwise the acis label does not show
    plt.show(block=False)

    # save
    filepath = os.path.join(filename)

    fig.set_size_inches(10, 8)
    plt.savefig(filepath, dpi=100)
    plt.close()


def plot_fitness(line_values, points_values, points_str):

    fig = plt.figure(figsize=(10, 8))

    x = np.linspace(0, 1, len(line_values))
    plt.plot(x, line_values)

    x = np.linspace(0, 1, len(points_values))
    plt.scatter(x, points_values)

    plt.xticks(x, points_str, rotation=90)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.4)

    return fig


def plot_fitness_from_model(info, v, variable_input_dim, n_points, n_lines, model, model_name, output_dim):

    points = sample_along_line(v, variable_input_dim, n_points)
    points_values = model.predict(points, output_dim)

    line_points = sample_along_line(v, variable_input_dim, n_lines)
    line_values = model.predict(line_points, output_dim)

    points_str = [i.__str__() for i in points]
    fig = plot_fitness(line_values, points_values, points_str)
    plt.title(model_name + '  :  ' + info['x_keys'].__str__())
    plt.ylabel(info['y_keys'][output_dim])

    return fig


def save_and_close_current_figure(fig, filename):
    # save
    filepath = os.path.join(filename)

    fig.set_size_inches(10, 8)
    plt.savefig(filepath, dpi=100)
    plt.close()


def plot_ternary_and_save(info, v, variable_input_dim, n_points, plot_dims, filename):
    points = sample_along_line(v, variable_input_dim, n_points)

    points_for_plot, axis_labels_for_plot = extract_info_for_plot(points, info['x_keys'], plot_dims)
    fig, tax = scatter_ternary_plot(points_for_plot, axis_labels_for_plot)
    save_and_close_current_ternary(fig, tax, filename)


if __name__ == '__main__':

    plt.close('all')

    #

    import datasets.tools
    x, y, info, path = datasets.tools.load_dataset('octanoic')

    # ["dep", "octanol", "octanoic", "pentanol"]
    dep_dim = info['x_keys'].index('dep')
    octanol_dim = info['x_keys'].index('octanol')
    octanoic_dim = info['x_keys'].index('octanoic')
    pentanol_dim = info['x_keys'].index('pentanol')

    # ["division", "directionality", "movement"]
    division_dim = info['y_keys'].index('division')
    directionality_dim = info['y_keys'].index('directionality')
    movement_dim = info['y_keys'].index('movement')

    #
    import models.regression.tools
    kernelridgerbf_file = os.path.join(root_path, 'models', 'regression', 'pickled', 'octanoic', 'KernelRidge-RBF.pkl')
    kernelridgerbf_model = models.regression.tools.load_model(kernelridgerbf_file)

    svrrbf_file = os.path.join(root_path, 'models', 'regression', 'pickled', 'octanoic', 'SVR-RBF.pkl')
    svrrbf_model = models.regression.tools.load_model(svrrbf_file)

    knn_file = os.path.join(root_path, 'models', 'regression', 'pickled', 'octanoic', 'KNeighborsRegressor.pkl')
    knn_model = models.regression.tools.load_model(knn_file)
    #
    n_points = 11
    n_lines = 101

    model = kernelridgerbf_model
    model_name = 'KernelRidge-RBF'

    #
    v = [-1] * 4
    v[octanol_dim] = 0
    v[pentanol_dim] = 0.2
    variable_input_dim = [dep_dim, octanoic_dim]

    for output_dim, output_name in enumerate(info['y_keys']):
        fig = plot_fitness_from_model(info, v, variable_input_dim, n_points, n_lines, model, model_name, output_dim)
        filename = os.path.join(HERE_PATH, 'traj1_{}.png'.format(output_name))
        save_and_close_current_figure(fig, filename)

    plot_dims = [dep_dim, octanoic_dim, pentanol_dim]
    filename = os.path.join(HERE_PATH, 'traj1_ternary.png')
    plot_ternary_and_save(info, v, variable_input_dim, n_points, plot_dims, filename)

    #
    v = [-1] * 4
    v[octanol_dim] = 0.15
    v[octanoic_dim] = 0
    variable_input_dim = [dep_dim, pentanol_dim]

    for output_dim, output_name in enumerate(info['y_keys']):
        fig = plot_fitness_from_model(info, v, variable_input_dim, n_points, n_lines, model, model_name, output_dim)
        filename = os.path.join(HERE_PATH, 'traj2_{}.png'.format(output_name))
        save_and_close_current_figure(fig, filename)

    plot_dims = [dep_dim, octanol_dim, pentanol_dim]
    filename = os.path.join(HERE_PATH, 'traj2_ternary.png')
    plot_ternary_and_save(info, v, variable_input_dim, n_points, plot_dims, filename)
