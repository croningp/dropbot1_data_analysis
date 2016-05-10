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

from utils import filetools


def sample_between_two_points(v_start, v_end, n_segment=10):
    """
    v_start, v_end are vectors, e.g. [0, 0.2, 0, 0.8]
    n is the number of segment, so n+1 point sampled
    """

    v_start = np.array(v_start)
    v_end = np.array(v_end)
    v = (v_end - v_start) / float(n_segment)

    out = []
    v_out = v_start
    for i in range(n_segment + 1):
        out.append(v_out)
        v_out = v_out + v

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
    # otherwise the axis label does not show
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


def plot_fitness_from_model(info, v_start, v_end, n_points, n_lines, model, model_name, output_dim):

    points = sample_between_two_points(v_start, v_end, n_points)
    points_values = model.predict(points, output_dim)

    line_points = sample_between_two_points(v_start, v_end, n_lines)
    line_values = model.predict(line_points, output_dim)

    points_str = [np.round(i, 2).__str__() for i in points]
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


def plot_ternary_and_save(info, v_start, v_end, n_points, plot_dims, filename):
    points = sample_between_two_points(v_start, v_end, n_points)

    points_for_plot, axis_labels_for_plot = extract_info_for_plot(points, info['x_keys'], plot_dims)
    fig, tax = scatter_ternary_plot(points_for_plot, axis_labels_for_plot)
    save_and_close_current_ternary(fig, tax, filename)


def plot_traj(foldername, v_start, v_end, plot_dims, n_points, n_lines, model, model_name):

    filetools.ensure_dir(foldername)

    for output_dim, output_name in enumerate(info['y_keys']):
        fig = plot_fitness_from_model(info, v_start, v_end, n_points, n_lines, model, model_name, output_dim)
        filename = os.path.join(foldername, 'traj_{}.png'.format(output_name))
        save_and_close_current_figure(fig, filename)

    filename = os.path.join(foldername, 'traj_ternary.png')
    plot_ternary_and_save(info, v_start, v_end, n_points, plot_dims, filename)


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
    # ["dep", "octanol", "octanoic", "pentanol"]
    v_start = [0, 0.15, 0, 0.85]
    v_end = [0.85, 0.15, 0, 0]
    plot_dims = [dep_dim, octanol_dim, pentanol_dim]
    foldername = os.path.join(HERE_PATH, 'traj1')

    plot_traj(foldername, v_start, v_end, plot_dims, n_points, n_lines, model, model_name)

    # ["dep", "octanol", "octanoic", "pentanol"]
    best_division = [0.782, 0.112, 0, 0.106]
    best_directionality = [0, 0.527, 0.242, 0.231]
    best_movement = [0.175, 0.802, 0.014, 0.009]

    v_start = best_division
    v_end = best_directionality
    plot_dims = [dep_dim, octanol_dim, pentanol_dim]
    foldername = os.path.join(HERE_PATH, 'traj2')

    plot_traj(foldername, v_start, v_end, plot_dims, n_points, n_lines, model, model_name)
