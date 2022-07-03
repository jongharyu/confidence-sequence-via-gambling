import numpy as np
import matplotlib.pyplot as plt

from methods.baselines import ConfidenceSeqeunce
from methods.lbup import LowerBoundUniversalPortfolioCI
from methods.up import UniversalPortfolioCI
from utils import confidence_interval


class HybridUPCI(ConfidenceSeqeunce):
    def __init__(self, delta, n=1, tup=50, betas=(1 / 2, 1 / 2)):
        super().__init__(delta)
        self.n = n  # the approximation order in LBUP
        self.tup = tup  # how long you will run UP at the beginning
        self.betas = betas  # UP parameter

    @confidence_interval
    def construct(self, xs, eps=0, tol=1e-5, verbose=False, log_every=100, **kwargs):
        lower_ci = np.zeros_like(xs).astype(float)
        upper_ci = np.zeros_like(xs).astype(float)

        # Run UP up until self.tup round
        lower_ci[:self.tup], upper_ci[:self.tup], telapsed_up, logweights = UniversalPortfolioCI(
            self.delta, betas=self.betas).construct(
            xs[:self.tup], eps=eps, tol=tol, verbose=verbose, log_every=log_every, do_not_apply_wor=True, **kwargs)

        # compute cumulative sums till t=self.tup which are to be used in the prior for LBUP
        sums0 = np.stack([(xs[:self.tup] ** k) for k in range(2 * self.n + 1)]).sum(axis=1)
        sums_c0 = np.stack([((1 - xs[:self.tup]) ** k) for k in range(2 * self.n + 1)]).sum(axis=1)

        # Run LBUP from then
        lower_ci[self.tup:], upper_ci[self.tup:], telapsed_lbup = LowerBoundUniversalPortfolioCI(
            self.delta, self.n,
            sums0=sums0, sums_c0=sums_c0,
            tup=self.tup, logweights=logweights).construct(
            xs[self.tup:], eps=eps, tol=tol, verbose=verbose, log_every=log_every, do_not_apply_wor=True, **kwargs)

        return lower_ci, upper_ci, np.array(telapsed_up + telapsed_lbup)

    def plot(self, xs, every=10, ax=None, legend=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)
        ms = np.arange(0.01, 1, 0.01)

        # Run UP up until self.tup round
        *_, logweights = UniversalPortfolioCI(self.delta, betas=self.betas).plot(
            xs[:self.tup], every, ax, legend, **kwargs)

        # compute cumulative sums till t=self.tup which are to be used in the prior for LBUP
        sums0 = np.stack([(xs[:self.tup] ** k) for k in range(2 * self.n + 1)]).sum(axis=1)
        sums_c0 = np.stack([((1 - xs[:self.tup]) ** k) for k in range(2 * self.n + 1)]).sum(axis=1)

        lbup = LowerBoundUniversalPortfolioCI(
            self.delta, self.n,
            sums0=sums0, sums_c0=sums_c0,
            tup=self.tup, logweights=logweights)

        sums = np.stack([(xs[self.tup:] ** k) for k in range(2 * self.n + 1)]).cumsum(axis=1).T  # (T, 2 * n + 1)
        sums_c = np.stack([((1 - xs[self.tup:]) ** k) for k in range(2 * self.n + 1)]).cumsum(axis=1).T  # (T, 2 * n + 1)

        # Run LBUP from then
        fs = []
        for t in range(self.tup + 1, len(xs) + 1):
            if t % every == 0:
                print(t)
                fs = np.zeros_like(ms)
                for i, m in enumerate(ms):
                    fs[i] = lbup.f(m, sums[t - self.tup - 1], sums_c[t - self.tup - 1])
                if 'label' not in kwargs:
                    kwargs['label'] = 'HybridUP'
                kwargs['label'] += ' (order={}; t={})'.format(self.n, t)
                ax.plot(ms, fs, **kwargs)
                ax.axhline(np.log(1 / self.delta), linestyle='--')
                if legend:
                    ax.legend()

        return fs
