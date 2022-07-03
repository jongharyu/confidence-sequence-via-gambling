import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betaln, logsumexp

from methods.baselines import ConfidenceSeqeunce
from utils import confidence_interval


class UniversalPortfolioCI(ConfidenceSeqeunce):
    def __init__(self, delta, betas=(1 / 2, 1 / 2)):
        super().__init__(delta)
        self.betas = betas

    def f(self, mu, t, logweights, eps=0, verbose=False):
        # log(wealth of Cover's UP)
        if verbose:
            print('t, mu:', t, mu)
        return logsumexp(logweights -
                         np.arange(t + 1) * np.log(mu + eps) -
                         (t - np.arange(t + 1)) * np.log(1 - mu + eps))

    def fprime(self, mu, t, logweights, eps=0):
        # derivative
        base = logweights - np.arange(t + 1) * np.log(mu + eps) - (t - np.arange(t + 1)) * np.log(1 - mu + eps)
        logdenominator = self.f(mu, t, logweights, eps, verbose=False)

        return np.exp(logsumexp(base[:-1] + np.log(t - np.arange(t)) - np.log(1 - mu + eps)) - logdenominator) - \
               np.exp(logsumexp(base[1:] + np.log(np.arange(1, t + 1)) - np.log(mu + eps)) - logdenominator)

    def update_logsumprod(self, logsumprod, x):
        if x == 0:
            logsumprod = logsumexp([np.pad(-np.inf * np.ones_like(logsumprod), (1, 0), constant_values=(-np.inf)),
                                    np.pad(logsumprod, (0, 1), constant_values=(-np.inf))],
                                   axis=0)
        elif x == 1:
            logsumprod = logsumexp([np.pad(logsumprod, (1, 0), constant_values=(-np.inf)),
                                    np.pad(-np.inf * np.ones_like(logsumprod), (0, 1), constant_values=(-np.inf))],
                                   axis=0)
        else:
            logsumprod = logsumexp([np.pad(logsumprod + np.log(x), (1, 0), constant_values=(-np.inf)),
                                    np.pad(logsumprod + np.log(1 - x), (0, 1), constant_values=(-np.inf))],
                                   axis=0)

        return logsumprod

    def compute_logweights(self, t, logsumprod):
        return logsumprod + betaln(np.arange(t + 1) + self.betas[0],
                                   t - np.arange(t + 1) + self.betas[1]) - betaln(*self.betas)

    @confidence_interval
    def construct(self, xs, eps=0, tol=1e-5, verbose=False, log_every=100, **kwargs):
        lower_ci = np.zeros_like(xs).astype(float)
        upper_ci = np.zeros_like(xs).astype(float)

        logsumprod = np.array([0.])

        xinit_low = 0.01
        xinit_up = 0.99

        telapsed = []
        start = time.time()
        for t in range(1, len(xs) + 1):
            x = xs[t - 1]
            logsumprod = self.update_logsumprod(logsumprod, x)
            logweights = self.compute_logweights(t, logsumprod)

            lower_ci[t - 1] = self.find_root(t, logweights,
                                             xinit=xinit_low, xmin=0, xmax=1,
                                             tol=tol, verbose=verbose)
            upper_ci[t - 1] = self.find_root(t, logweights,
                                             xinit=xinit_up, xmin=0, xmax=1,
                                             tol=tol, verbose=verbose)

            xinit_low = lower_ci[t - 1] if not np.isnan(lower_ci[t - 1]) else 1e-6
            xinit_up = upper_ci[t - 1] if not np.isnan(upper_ci[t - 1]) else 1 - 1e-6

            if t % log_every == 0:
                end = time.time()
                telapsed.append(end - start)
                print(t, end=' ')
                start = end

        return lower_ci, upper_ci, telapsed, logweights

    def plot(self, xs, every=10, ax=None, legend=False, **kwargs):
        xs = np.atleast_1d(xs.squeeze())
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)
        ms = np.arange(0.01, 1, 0.01)

        fs = []
        fps = []
        logsumprod = np.array([0.])
        for t in range(1, len(xs) + 1):
            x = xs[t - 1]
            logsumprod = self.update_logsumprod(logsumprod, x)
            logweights = self.compute_logweights(t, logsumprod)

            if t % every == 0:
                print(t, end=' ')
                fs = np.zeros_like(ms)
                fps = np.zeros_like(ms)
                for i, m in enumerate(ms):
                    fs[i] = self.f(m, t, logweights)
                    fps[i] = self.fprime(m, t, logweights)
                if 'label' not in kwargs:
                    kwargs['label'] = 'UP'
                kwargs['label'] += ' (t={})'.format(t)
                ax.plot(ms, fs, **kwargs)
                ax.axhline(np.log(1 / self.delta), linestyle='--')
                if legend:
                    ax.legend()

        return fs, fps, logweights
