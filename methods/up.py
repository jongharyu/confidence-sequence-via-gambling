import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betaln, logsumexp
from tqdm import tqdm

from methods.base import ConfidenceSequence
from utils import confidence_interval, multibetaln


class StockInvestmentCI(ConfidenceSequence):
    def __init__(self, betas=(1 / 2, 1 / 2)):
        super().__init__()
        self.betas = betas

    def f(self, m, t, logweights, eps=0, verbose=False):
        # log(wealth of Cover's UP)
        if verbose:
            print('t, mu:', t, m)
        return logsumexp(logweights -
                         (np.arange(t + 1) * np.log(m + eps) +
                          (t - np.arange(t + 1)) * np.log(1 - m + eps)))

    def fprime(self, mu, t, logweights, eps=0):
        # derivative
        base = logweights - np.arange(t + 1) * np.log(mu + eps) - (t - np.arange(t + 1)) * np.log(1 - mu + eps)
        log_denom = self.f(mu, t, logweights, eps, verbose=False)

        return np.exp(logsumexp(base[:-1] + np.log(t - np.arange(t)) - np.log(1 - mu + eps)) - log_denom) - \
               np.exp(logsumexp(base[1:] + np.log(np.arange(1, t + 1)) - np.log(mu + eps)) - log_denom)

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
        return logsumprod + \
               (betaln(np.arange(t + 1) + self.betas[0], t - np.arange(t + 1) + self.betas[1]) -
                betaln(*self.betas))

    @confidence_interval
    def construct(self, delta, xs, eps=0, tol=1e-5, verbose=False, log_every=100, **kwargs):
        lower_ci = np.zeros_like(xs).astype(float)
        upper_ci = np.zeros_like(xs).astype(float)

        logsumprod = np.array([0.])
        logweights = np.array([0.])

        xinit_low = 0.01
        xinit_up = 0.99

        telapsed = []
        start = time.time()
        for t in tqdm(range(1, len(xs) + 1)):
            x = xs[t - 1]
            logsumprod = self.update_logsumprod(logsumprod, x)
            logweights = self.compute_logweights(t, logsumprod)

            if verbose:
                # to see if log wealth(mu_hat) <= 0 always:
                mu_hat = xs[:t].mean()
                f_mu_hat = self.f(mu_hat, t, logweights)
                if f_mu_hat >= 0:
                    print("t={}, mu_hat={}, f_t(mu_hat)={}".format(t, mu_hat, f_mu_hat))
                    print("t={}, mu_hat={}, f_t'(mu_hat)={}".format(t, mu_hat, self.fprime(mu_hat, t, logweights)))

            lower_ci[t - 1] = self.find_root(delta, t, logweights,
                                             xinit=xinit_low, xmin=0, xmax=1,
                                             tol=tol, verbose=verbose)
            upper_ci[t - 1] = self.find_root(delta, t, logweights,
                                             xinit=xinit_up, xmin=0, xmax=1,
                                             tol=tol, verbose=verbose)

            xinit_low = lower_ci[t - 1] if not np.isnan(lower_ci[t - 1]) else 1e-6
            xinit_up = upper_ci[t - 1] if not np.isnan(upper_ci[t - 1]) else 1 - 1e-6

            if t % log_every == 0:
                end = time.time()
                telapsed.append(end - start)
                # print(t, end=' ')
                start = end

        return lower_ci, upper_ci, telapsed, logweights

    def plot(self, delta, xs, every=10, ax=None, legend=False, **kwargs):
        xs = np.atleast_1d(xs.squeeze())
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)
        ms = np.arange(0.01, 1, 0.01)

        fs = []
        fps = []
        logsumprod = np.array([0.])
        logweights = np.array([0.])
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
                ax.plot(ms, fs, **kwargs)
                ax.axhline(np.log(1 / delta), linestyle='--')
                if legend:
                    ax.legend()

        return fs, fps, logweights


class MultiStockInvestmentCI(ConfidenceSequence):
    def __init__(self, M=2, betas=None):
        super().__init__()
        self.M = M
        self.betas = .5 * np.ones((self.M,)) if betas is None else np.array(betas)
        self.logsumprod = None

    def compute_logsumprod_batch(self, ys, verbose=False):
        logsumprod = dict()
        logsumprod[tuple(np.zeros((self.M,)))] = 0
        for t in range(1, ys.shape[1] + 1):
            yv = ys[:, t - 1]
            logsumprod = self.update_logsumprod(logsumprod, yv)
            if verbose:
                print(t, end=' ')
        self.logsumprod = logsumprod
        return logsumprod

    def clean_logsumprod(self):
        print('logsumprod had length {}'.format(len(self.logsumprod)), end=', ')
        for kv in list(self.logsumprod.keys()):
            if self.logsumprod[kv] == -np.inf:
                del self.logsumprod[kv]
        print('and is cut to {}'.format(len(self.logsumprod)))

    def f(self, m, eps=0, verbose=False):
        # note: unlike in k=2 case, logsumprod is given here
        # note:
        #   logweights[kv] = multibetaln(kv + self.betas) - multibetaln(self.betas) + logsumprod[kv]
        # m: (n, M)
        return logsumexp(np.stack([- (np.array(kv) * np.log(m)).sum(axis=-1) +
                                   multibetaln(kv + self.betas) - multibetaln(self.betas) +
                                   self.logsumprod[kv] for kv in self.logsumprod], axis=-1),
                         axis=-1)  # (n, )

    def update_logsumprod(self, logsumprod, yv):
        logsumprod_next = defaultdict(list)
        for kv in logsumprod:
            for j in range(self.M):
                logsumprod_next[tuple(np.array(kv) + standard_vector(j, self.M))].append(
                    logsumprod[kv] + np.log(yv[j]))
        for kv in logsumprod_next:
            logsumprod_next[kv] = logsumexp(logsumprod_next[kv], axis=0)

        return logsumprod_next


def standard_vector(j, M):
    tmp = np.zeros((M, ))
    tmp[j] = 1
    return tmp
