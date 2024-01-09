import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from methods.base import ConfidenceSequence
from methods.lbup import LowerBoundStockInvestmentCI
from methods.up import StockInvestmentCI
from utils import confidence_interval


class HybridCI(ConfidenceSequence):
    def __init__(self, n=1, tup=50, betas=(1 / 2, 1 / 2)):
        super().__init__()
        self.n = n  # the approximation order in LBUP
        self.tup = tup  # how long you will run UP at the beginning
        self.betas = betas  # UP parameter

    @confidence_interval
    def construct(self, delta, xs, eps=0, tol=1e-5, verbose=False, log_every=100, **kwargs):
        lower_ci = np.zeros_like(xs).astype(float)
        upper_ci = np.zeros_like(xs).astype(float)

        # Run UP up until self.tup round
        lower_ci[:self.tup], upper_ci[:self.tup], telapsed_up, logweights = StockInvestmentCI(
            betas=self.betas).construct(
            delta, xs[:self.tup],
            eps=eps, tol=tol, verbose=verbose, log_every=log_every, do_not_apply_wor=True, **kwargs)

        # compute cumulative sums till t=self.tup which are to be used in the prior for LBUP
        sums0 = np.stack([(xs[:self.tup] ** k) for k in range(2 * self.n + 1)]).sum(axis=1)
        sums_c0 = np.stack([((1 - xs[:self.tup]) ** k) for k in range(2 * self.n + 1)]).sum(axis=1)

        # Run LBUP from then
        lower_ci[self.tup:], upper_ci[self.tup:], telapsed_lbup = LowerBoundStockInvestmentCI(
            self.n,
            sums0=sums0, sums_c0=sums_c0,
            tup=self.tup, logweights=logweights).construct(
            delta, xs[self.tup:],
            eps=eps, tol=tol, verbose=verbose, log_every=log_every, do_not_apply_wor=True, **kwargs)

        return lower_ci, upper_ci, np.array(telapsed_up + telapsed_lbup)

    def plot(self, delta, xs, every=10, ax=None, legend=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)
        ms = np.arange(0.01, 1, 0.01)

        # Run UP up until self.tup round
        *_, logweights = StockInvestmentCI(betas=self.betas).plot(
            delta, xs[:self.tup], every, ax, legend, **kwargs)

        # compute cumulative sums till t=self.tup which are to be used in the prior for LBUP
        sums0 = np.stack([(xs[:self.tup] ** k) for k in range(2 * self.n + 1)]).sum(axis=1)
        sums_c0 = np.stack([((1 - xs[:self.tup]) ** k) for k in range(2 * self.n + 1)]).sum(axis=1)

        lbup = LowerBoundStockInvestmentCI(
            self.n,
            sums0=sums0, sums_c0=sums_c0,
            tup=self.tup, logweights=logweights)

        sums = np.stack([(xs[self.tup:] ** k) for k in range(2 * self.n + 1)]).cumsum(axis=1).T  # (T, 2 * n + 1)
        sums_c = np.stack([((1 - xs[self.tup:]) ** k) for k in range(2 * self.n + 1)]).cumsum(axis=1).T  # (T, 2 * n + 1)

        # Run LBUP from then
        fs = []
        for t in tqdm(range(self.tup + 1, len(xs) + 1)):
            if t % every == 0:
                mu_hat = (sums[t - self.tup - 1, 1] + sums0[1]) / (sums[t - self.tup - 1, 0] + sums0[0])
                print("t={}, f(mu_hat)={}".format(t + self.tup, lbup.f(mu_hat, sums[t - self.tup - 1], sums_c[t - self.tup - 1])))
                fs = np.zeros_like(ms)
                for i, m in enumerate(ms):
                    fs[i] = lbup.f(m, sums[t - self.tup - 1], sums_c[t - self.tup - 1])
                if 'label' not in kwargs:
                    kwargs['label'] = 'HybridUP'
                kwargs['label'] += ' (order={}; t={})'.format(self.n, t)
                ax.plot(ms, fs, **kwargs)
                ax.axhline(np.log(1 / delta), linestyle='--')
                ax.axvline(x=mu_hat)
                if legend:
                    ax.legend()

        return fs
