import os
import json
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# adding parent directory to path, so we can access the utils easily
import sys
root_path = os.path.join(HERE_PATH, '..', '..')
sys.path.append(root_path)

from utils import filetools

from datasets.tools import load_dataset

if __name__ == '__main__':

    import itertools
    from scipy.stats.stats import pearsonr

    x, y, info, path = load_dataset('octanoic')

    save_path = os.path.join(HERE_PATH, 'plot')
    filetools.ensure_dir(save_path)

    for comb in itertools.combinations(range(y.shape[1]), 2):
        id1 = comb[0]
        name1 = info['y_keys'][id1]

        id2 = comb[1]
        name2 = info['y_keys'][id2]

        print 'Correlation between {} and {}'.format(name1, name2)
        r, p = pearsonr(y[:, id1], y[:, id2])
        print 'r = {}, p = {}'.format(r, p)

        seaborn.set(font_scale=2)
        h = seaborn.jointplot(y[:, id1], y[:, id2], kind='scatter', size=10)
        h.ax_joint.set_xlabel(name1)
        h.ax_joint.set_ylabel(name2)
        plt.tight_layout()

        for ext in ['png', 'eps', 'svg']:
            filename = '{}_{}.{}'.format(name1, name2, ext)
            filepath = os.path.join(save_path, filename)
            plt.savefig(filepath, dpi=100)

        plt.close()
