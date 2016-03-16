from sklearn.mixture import GMM

import gmr


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
