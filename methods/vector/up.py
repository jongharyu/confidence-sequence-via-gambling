from collections import defaultdict

import numpy as np
from scipy.special import logsumexp

from methods.scalar.base import ConfidenceSequence
from methods.scalar.up import UniversalPortfolioCS
from utils.special_functions import multibetaln


# Two-stock universal portfolio combined via average of wealths for multidim case
class CombinedUniversalPortfolioCS:
    def __init__(self, M=2, betas=(1 / 2, 1 / 2)):
        self.M = M
        self.logsumprods = [np.array([0.0]) for _ in range(self.M)]
        self.logweights = [0.0 for _ in range(self.M)]
        self.ups = [UniversalPortfolioCS(betas=betas) for _ in range(self.M)]

    def f(self, m, t, eps=0, verbose=False):
        # m: (n, M)
        log_wealth = logsumexp(
            np.stack(
                [
                    self.fbase(m[:, i], t, self.logweights[i], eps)
                    for i in range(self.M)
                ],
                axis=0,
            ),
            axis=0,
        ) - np.log(self.M)
        return log_wealth

    def update_logsumprods(self, t, yv):
        for i in range(self.M):
            self.logsumprods[i] = self.ups[i].update_logsumprod(
                self.logsumprods[i], yv[i]
            )
            self.logweights[i] = self.ups[i].compute_logweights(t, self.logsumprods[i])

    def fbase(self, m, t, logweights, eps=0, verbose=False):
        """
        m: (n, )
        t: int
        logweights: (l, )
        """
        # log(wealth of Cover's UP)
        if verbose:
            print("t, m:", t, m)
        m = m.reshape(-1, 1)  # (n, 1)
        logweights = logweights.reshape(1, -1)  # (1, l)
        return logsumexp(
            logweights
            - np.arange(t + 1) * np.log(m + eps)
            - (t - np.arange(t + 1)) * np.log(1 - m + eps),
            axis=-1,
        )  # (n, )


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
        # m: (n, M)
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
        )  # (n, )

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
