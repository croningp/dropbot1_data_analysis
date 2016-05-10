import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
import ternary

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..', '..')
sys.path.append(root_path)

from utils import filetools


def load_model(filename):
    return pickle.load(open(filename))


def plot_ternary(clf, nCompound, combination, axis_labels, scale=50, vmin=None, vmax=None, fontsize=22):

    # function called by ternary to estimate the value at each point
    def compute_value(s):
        x_ternary = np.zeros((nCompound,))
        for i, coumpound_idx in enumerate(combination):
            x_ternary[coumpound_idx] = s[i]

        value = clf.predict(x_ternary)[0]
        if vmin is not None or vmax is not None:
            value = np.clip(value, vmin, vmax)
        return value

    fig, tax = ternary.figure(scale=scale)
    tax.heatmapf(compute_value, boundary=True, style="triangular", scientific=True, vmin=vmin, vmax=vmax)

    # make the plot look good
    tax.boundary(color="black")
    # tax.gridlines(multiple=scale / 10.0, color="black")

    # Set Axis labels and Title
    tax.right_axis_label(axis_labels[0], fontsize=fontsize)
    tax.left_axis_label(axis_labels[1], fontsize=fontsize)
    tax.bottom_axis_label(axis_labels[2], fontsize=fontsize)

    ticks = list(np.linspace(0, 1, 11))
    loc = list(np.linspace(0, scale, 11))
    tax.ticks(ticks=ticks, locations=loc, axis='lbr', linewidth=1, multiple=5, color="black")

    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    fig.patch.set_visible(False)

    return fig, tax


def save_and_close_ternary(fig, tax, filebasename, exts=['.png', '.eps', '.svg']):
    # weird stuff to make it save the plot properly
    # otherwise the axis label does not show
    plt.show(block=False)

    for ext in exts:
        # save
        filepath = filebasename + ext

        fig.set_size_inches(12, 8)
        plt.savefig(filepath, dpi=100)

    plt.close()


if __name__ == '__main__':

    kernelridgerbf_file = os.path.join(root_path, 'models', 'regression', 'pickled', 'octanoic', 'KernelRidge-RBF.pkl')
    clfs = load_model(kernelridgerbf_file)

    oil_names = ['DEP', "1-Octanol", "Octanoic acid", "1-Pentanol"]
    combination = [0, 1, 3]
    axis_labels = [oil_names[i] for i in combination]

    save_folder = os.path.join(HERE_PATH, 'plot')
    filetools.ensure_dir(save_folder)

    scale = 100
    fontsize = 22

    # ["division", "directionality", "movement"]
    # division
    fig, tax = plot_ternary(clfs[0], 4, combination, axis_labels, scale=scale, vmin=0, vmax=15, fontsize=fontsize)
    plt.title('Division landscape', fontsize=fontsize)
    filebasename = os.path.join(save_folder, 'division_ternary')
    save_and_close_ternary(fig, tax, filebasename)

    # directionality
    fig, tax = plot_ternary(clfs[1], 4, combination, axis_labels, scale=scale, vmin=0, vmax=6, fontsize=fontsize)
    plt.title('Directionality landscape', fontsize=fontsize)
    filebasename = os.path.join(save_folder, 'directionality_ternary')
    save_and_close_ternary(fig, tax, filebasename)
