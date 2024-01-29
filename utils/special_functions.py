import numpy as np
from scipy.special import gammaln


def binary_entropy(p):
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def multibetaln(alphas):
    return gammaln(alphas).sum(axis=0) - gammaln(alphas.sum(axis=0))


def multinomln(ns, axis=-1):
    # ns: (n, M) by default
    mask = ((ns < 0).sum(axis=axis)).astype(bool).astype(float)  # (n, )
    mask[mask == 1.] = -np.inf
    # print("ns", ns, "mask", mask)
    return mask + (gammaln(ns.sum(axis=axis) + 1) - gammaln(ns + 1).sum(axis=axis))  # (n, )
