import numpy as np
from scipy.special import logsumexp

from methods.scalar.onesided import (
    UnboundedUniversalPortfolioCS,
    UnboundedLowerBoundUniversalPortfolioCS,
)
from methods.scalar.up import UniversalPortfolioCS


# Two-stock universal portfolio combined via average of wealths for multidim case
class MultivariateCombinedUniversalPortfolioCS:
    def __init__(self, M, betas=(1 / 2, 1 / 2)):
        self.M = M  # number of stocks
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


class MultivariateCombinedUnboundedUniversalPortfolioCS:
    def __init__(self, M, betas=(1 / 2, 1 / 2)):
        self.M = M  # number of stocks
        self.logsumprods = [np.array([0.0]) for _ in range(self.M)]
        self.logweights = [0.0 for _ in range(self.M)]
        self.ups = [UnboundedUniversalPortfolioCS(betas=betas) for _ in range(self.M)]

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
            logweights - np.arange(t + 1) * np.log(m + eps), axis=-1
        )  # (n, )


class MultivariateCombinedUnboundedLowerBoundUniversalPortfolioCS:
    def __init__(self, M, n, tup=0, betas=(1 / 2, 1 / 2), logweights=None):
        self.M = M
        self.n = n  # the approximation order in LBUP

        # for piggybacking UP (these are used in HybridUP)
        self.tup = tup

        self.logsumprods = [np.array([0.0]) for _ in range(self.M)]
        self.logweights = (
            [None for _ in range(self.M)] if logweights is None else logweights
        )
        self.lbups = [
            UnboundedLowerBoundUniversalPortfolioCS(
                n=n, betas=betas, logweights=self.logweights[i]
            )
            for i in range(self.M)
        ]

    def f(self, m, sums):
        # m: (n_grid, M)
        # sums: (M, 2 * n + 1)
        log_wealths = np.stack(
            [self.lbups[i].f(m[:, i], sums[i]) for i in range(self.M)], axis=0
        )  # (M, n_grid)
        print("DEUBGGING", log_wealths, log_wealths.shape)
        log_wealth = logsumexp(np.stack(log_wealths, axis=0), axis=0) - np.log(self.M)
        return log_wealth


class MultivariateCombinedUnboundedHybridUniversalPortfolioCS:
    def __init__(self, M, n=1, tup=50, betas=(1 / 2, 1 / 2)):
        self.M = M  # number of stocks
        self.n = n  # the approximation order in LBUP
        self.tup = tup  # how long you will run UP at the beginning

        self.logsumprods = [np.array([0.0]) for _ in range(self.M)]
        self.logweights = [np.array([0.0]) for _ in range(self.M)]
        self.ups = [UnboundedUniversalPortfolioCS(betas=betas) for i in range(self.M)]
        self.lbups = [
            UnboundedLowerBoundUniversalPortfolioCS(
                n=n, tup=tup, betas=betas, logweights=None
            )
            for i in range(self.M)
        ]

    def update_stats(self, t, x):
        # x: (M, )
        if t < self.tup:
            # update UnboundedLowerBoundUniversalPortfolioCS
            for i in range(self.M):
                self.ups[i].update_logsumprod(self.ups[i].logsumprod, x[i])
                self.logweights[i] = self.ups[i].compute_logweights(
                    t, self.ups[i].logsumprod
                )
        else:
            # update UnboundedLowerBoundUniversalPortfolioCS
            for i in range(self.M):
                if t == self.tup:
                    self.lbups[i].logweights = self.ups[i].compute_logweights(
                        t, self.ups[i].logsumprod
                    )
                self.lbups[i].update_sums(x[i])

    def f_sequential(self, m):
        # m: (n_grid, M)
        log_wealth = logsumexp(
            np.stack(
                [self.lbups[i].f_sequential(m[:, i]) for i in range(self.M)],
                axis=0,
            ),
            axis=0,
        ) - np.log(self.M)
        return log_wealth
