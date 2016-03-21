import pickle
import numpy as np

from sklearn.mixture import GMM

import gmr


def load_model(filename):
    gmms = pickle.load(open(filename))
    gmminf = sklearn_to_gmr(gmms)
    return Model(gmminf)


# convert a sklearn GMM into a gmr GMM
# gmr allows for conditional inference, useful for querying the GMM
def sklearn_to_gmr(gmms):
    if type(gmms) == list:
        gmminf = []
        for gmm in gmms:
            n_components = gmm.weights_.shape[0]
            gmminf.append(gmr.gmm.GMM(n_components,
                                      gmm.weights_,
                                      gmm.means_,
                                      gmm.covars_))
    elif type(gmms) == GMM:
        gmm = gmms
        n_components = gmm.weights_.shape[0]
        gmminf = gmr.gmm.GMM(n_components,
                             gmm.weights_,
                             gmm.means_,
                             gmm.covars_)

    return gmminf


class Model(object):

    def __init__(self, gmminf):
        self.model = gmminf

    def is_model_a_list(self):
        return type(self.model) == list

    def n_out_dims(self, n_in_dims):

        if self.is_model_a_list():
            return len(self.model)
        else:
            total_dims = self.model.means.shape[1]
            return total_dims - n_in_dims

    def predict(self, x, out_dims=None):
        x = np.atleast_2d(x)
        in_dims = range(x.shape[1])

        if out_dims is None:
            n_in_dims = len(in_dims)
            out_dims = range(self.n_out_dims(n_in_dims))
        else:
            if type(out_dims) == int:
                out_dims = [out_dims]

        output = np.zeros((x.shape[0], len(out_dims)))
        if self.is_model_a_list():
            for i, row in enumerate(x):
                for j, dim in enumerate(out_dims):
                    cond = self.model[dim].condition(in_dims, row)
                    output[i, j] = cond.sample(1)
        else:
            for i, row in enumerate(x):
                cond = self.model.condition(in_dims, row)
                sample = cond.sample(1)
                for j, dim in enumerate(out_dims):
                    output[i, j] = sample[0, dim]

        return output
