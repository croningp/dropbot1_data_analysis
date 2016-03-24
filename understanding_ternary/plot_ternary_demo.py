import os
import numpy as np

import matplotlib.pyplot as plt
import ternary

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


if __name__ == '__main__':

    fontsize = 22

    fig, tax = ternary.figure(scale=1.0)

    x = [[0.1, 0.2, 0.7]]
    tax.scatter(x, marker='D', color='g', label=x.__str__())

    x = [[0.1, 0, 0.9]]
    tax.scatter(x, marker='s', color='r', label=x.__str__())

    x = [[0.3, 0.2, 0.5]]
    tax.scatter(x, marker='o', color='b', label=x.__str__())

    x = [[0.4, 0.4, 0.2]]
    tax.scatter(x, marker='h', color='m', label=x.__str__())

    # make the plot look good
    tax.boundary()
    tax.gridlines(multiple=0.1, color="black")
    tax.legend()

    # Set Axis labels and Title
    tax.right_axis_label('A', fontsize=fontsize)
    tax.left_axis_label('B', fontsize=fontsize)
    tax.bottom_axis_label('C', fontsize=fontsize)

    ticks = list(np.linspace(0, 1, 11))
    loc = list(np.linspace(0, 1, 11))
    tax.ticks(ticks=ticks, locations=loc, axis='lbr', linewidth=1, multiple=5)

    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    fig.patch.set_visible(False)

    # weird stuff to make it save the plot properly
    # otherwise the acis label does not show
    plt.show(block=False)

    # save
    filename = "key_ternary_points.png"
    filepath = os.path.join(HERE_PATH, filename)

    fig.set_size_inches(10, 8)
    plt.savefig(filepath, dpi=100)
    plt.close()
