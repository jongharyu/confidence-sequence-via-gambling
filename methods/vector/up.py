from collections import defaultdict

import numpy as np
from scipy.special import logsumexp

from methods.scalar.base import ConfidenceSequence
from utils.special_functions import multibetaln


class MultivariateUniversalPortfolioCS(ConfidenceSequence):
    def __init__(self, M=2, betas=None):
        super().__init__()
        self.M = M
        self.betas = 0.5 * np.ones((self.M,)) if betas is None else np.array(betas)
        self.logsumprod = self._init_logsumprod()

    def _init_logsumprod(self):
        logsumprod = dict()
        logsumprod[tuple(np.zeros((self.M,)))] = 0
        return logsumprod

    def update_logsumprod_batch(self, ys, verbose=False):
        logsumprod = self._init_logsumprod()
        for t in range(1, ys.shape[1] + 1):
            yv = ys[:, t - 1]
            logsumprod = self._update_logsumprod(yv, logsumprod)
            if verbose:
                print(t, end=" ")
        self.logsumprod = logsumprod

    def clean_logsumprod(self):
        print("logsumprod had length {}".format(len(self.logsumprod)), end=", ")
        for kv in list(self.logsumprod.keys()):
            if self.logsumprod[kv] == -np.inf:
                del self.logsumprod[kv]
        print("and is cut to {}".format(len(self.logsumprod)))

    def f(self, m, eps=0, verbose=False):
        # note: unlike in k=2 case, logsumprod is given here
        # note:
        #   logweights[kv] = multibetaln(kv + self.betas) - multibetaln(self.betas) + logsumprod[kv]
        # m: (n_grid, M)
        return logsumexp(
            np.stack(
                [
                    -(np.array(kv) * np.log(m)).sum(axis=-1)
                    + multibetaln(kv + self.betas)
                    - multibetaln(self.betas)
                    + self.logsumprod[kv]
                    for kv in self.logsumprod
                ],
                axis=-1,
            ),
            axis=-1,
        )  # (n_grid, )

    def update_logsumprod(self, yv):
        self.logsumprod = self._update_logsumprod(yv, self.logsumprod)

    def _update_logsumprod(self, yv, logsumprod):
        logsumprod_next = defaultdict(list)
        for kv in logsumprod:
            for j in range(self.M):
                logsumprod_next[
                    tuple(np.array(kv) + standard_vector(j, self.M))
                ].append(logsumprod[kv] + np.log(yv[j]))
        for kv in logsumprod_next:
            logsumprod_next[kv] = logsumexp(logsumprod_next[kv], axis=0)
        return logsumprod_next


def standard_vector(j, M):
    tmp = np.zeros((M,))
    tmp[j] = 1
    return tmp
