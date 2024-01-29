import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import betaln, digamma
from tqdm import tqdm

from methods.base import ConfidenceSequence, confidence_interval
from utils.special_functions import binary_entropy, multibetaln


# based on discrete-coin betting + "naive embedding"
class CoinBettingCI(ConfidenceSequence):
    def __init__(self):
        super().__init__()

    def f(self, x, t):
        return t * np.log(2) + betaln((t + x + 1) / 2, (t - x + 1) / 2) - betaln(1 / 2, 1 / 2)

    def fprime(self, x, t):
        return 1 / 2 * (digamma((t + x + 1) / 2) - digamma((t - x + 1) / 2))

    @confidence_interval
    def construct(self, delta, xs, verbose=False, **kwargs):
        ts = np.arange(1, len(xs) + 1)
        mu_hats = xs.cumsum() / ts
        roots = self.find_root(delta, ts, xmin=-ts, xmax=ts, verbose=verbose)

        lower_ci = mu_hats - roots / ts
        upper_ci = mu_hats + roots / ts

        return lower_ci, upper_ci

    def plot(self, xs, every=10, ax=None, legend=False, **kwargs):
        raise NotImplementedError


# based on discrete-coin betting + "tighter embedding"
class HorseRaceCI(ConfidenceSequence):
    def __init__(self, betas=(1 / 2, 1 / 2)):
        super().__init__()
        self.betas = betas

    def f(self, m, t, s, eps=0):
        # negative log bernoulli probability with count s at time step t
        return - s * np.log(m + eps) - (t - s) * np.log(1 - m + eps) \
               + betaln(s + self.betas[0], t - s + self.betas[1]) - betaln(*self.betas)

    def fprime(self, m, t, s, eps=0):
        # derivative
        return - s / (m + eps) + (t - s) / (1 - m + eps)

    @confidence_interval
    def construct(self, delta, xs, eps=1e-3, tol=1e-5, verbose=False, batch=False, log_every=100, tqdm_=True, **kwargs):
        tqdm_ = tqdm if tqdm_ else lambda x: x
        ts = np.arange(1, len(xs) + 1)
        ss = xs.cumsum()
        telapsed = []

        if batch:
            lower_ci = self.find_root(delta, ts, ss, xinit=eps, xmin=0, xmax=1, verbose=verbose)
            upper_ci = self.find_root(delta, ts, ss, xinit=1 - eps, xmin=0, xmax=1, verbose=verbose)
        else:
            lower_ci = np.zeros_like(xs).astype(float)
            upper_ci = np.ones_like(xs).astype(float)

            start = time.time()
            for t in tqdm_(range(1, len(xs) + 1)):
                mu_hat = ss[t - 1] / t
                if mu_hat == 0:
                    mu_hat = eps
                if mu_hat == 1:
                    mu_hat = 1 - eps

                # use scipy's bisect with a customized initialization rule
                xinit_low = lower_ci[t - 2] if t > 1 else eps
                if self.f(xinit_low, t, ss[t - 1]) < np.log(1 / delta):
                    lower_ci[t - 1] = lower_ci[t - 2]
                else:
                    lower_ci[t - 1] = self.find_root_bisect(delta, t, ss[t - 1],
                                                            xinits=(lower_ci[t - 2], mu_hat),
                                                            tol=tol,
                                                            verbose=verbose)
                    if lower_ci[t - 1] == -1:
                        print("bisect encounters ValueError!")
                        lower_ci[t - 1] = lower_ci[t - 2]

                xinit_up = upper_ci[t - 2] if t > 1 else 1 - eps
                if self.f(xinit_up, t, ss[t - 1]) < np.log(1 / delta):
                    upper_ci[t - 1] = upper_ci[t - 2]
                else:
                    upper_ci[t - 1] = self.find_root_bisect(delta, t, ss[t - 1],
                                                            xinits=(mu_hat, upper_ci[t - 2]),
                                                            tol=tol,
                                                            verbose=verbose)
                    if upper_ci[t - 1] == -1:
                        print("bisect encounters ValueError!")
                        upper_ci[t - 1] = upper_ci[t - 2]

                if t % log_every == 0:
                    end = time.time()
                    telapsed.append(end - start)
                    start = end

        return lower_ci, upper_ci, telapsed

    @confidence_interval
    def construct_outer(self, delta, xs):
        ts = np.arange(1, len(xs) + 1)
        ss = xs.cumsum()

        mu_hats = ss / ts
        logqkt = betaln(ss + self.betas[0], ts - ss + self.betas[1]) - betaln(*self.betas)
        gs = 1 / ts * (np.log(1 / delta) - logqkt) - binary_entropy(mu_hats)

        lower_ci = mu_hats - np.sqrt(gs / 2)
        upper_ci = mu_hats + np.sqrt(gs / 2)

        return lower_ci, upper_ci

    def plot(self, delta, xs, every=10, ax=None, legend=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)
        mus = np.arange(0.01, 1, 0.01)

        fs = []
        for t in tqdm(range(1, len(xs) + 1)):
            if t % every == 0:
                fs = self.f(mus, t, xs[:t].sum())
                if 'label' not in kwargs:
                    kwargs['label'] = 'HR'
                ax.plot(mus, fs, **kwargs)
                ax.axhline(np.log(1 / delta), linestyle='--')
                if legend:
                    ax.legend()

        return fs


class UnboundedHorseRaceCI(HorseRaceCI):
    def f(self, m, t, xs, eps=0):
        cs = np.maximum.accumulate(xs)
        zs = xs / cs
        log_odd_term = (
                zs * np.nan_to_num(np.log(np.float64(1.) / (m / cs)), nan=0., posinf=0.) +
                (1 - zs) * np.nan_to_num(np.log(np.float64(1.) / (1 - np.minimum(m / cs, np.ones_like(xs)))), nan=0., posinf=0.)
        ).sum()
        log_prob = betaln(zs.sum() + self.betas[0],
                          (1 - zs).sum() + self.betas[1]) - \
                   betaln(*self.betas)

        return log_odd_term + log_prob

    def fprime(self, m, t, s, eps=0):
        raise NotImplementedError

    @confidence_interval
    def construct(self, xs, eps=1e-3, tol=1e-5, verbose=False, batch=False, log_every=100, **kwargs):
        raise NotImplementedError

    def plot(self, delta, xs, upper_bound=1, every=10, ax=None, legend=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)
        ms = np.arange(0.01, upper_bound, 0.005)

        fs = []
        for t in tqdm(range(1, len(xs) + 1)):
            if t % every == 0:
                fs = np.array([self.f(m, t, xs[:t]) for m in ms])
                # fs[fs == np.inf] = 1e3
                if 'label' not in kwargs:
                    kwargs['label'] = 'UnbddKT'
                cummax = np.maximum.accumulate(xs[:t])[-1]
                ax.plot(ms, fs, **kwargs)
                ax.axhline(np.log(1 / delta), linestyle='--')
                ax.axvline(cummax, linestyle='--', c='red')
                if legend:
                    ax.legend()

        # print(len(xs), self.f(1., len(xs), xs))
        # print(list(zip(ms, fs)))
        return fs


class TruncatedHorseRaceCI(HorseRaceCI):
    def f(self, m, ct, t, xs, eps=0):
        zs = np.minimum(xs / ct, np.ones_like(xs))
        log_odd_term = (
                zs * np.nan_to_num(np.log(np.float64(1.) / m), nan=0., posinf=0.) +
                (1 - zs) * np.nan_to_num(np.log(np.float64(1.) / (1 - m)), nan=0., posinf=0.)
        ).sum()
        log_prob = betaln(zs.sum() + self.betas[0],
                          (1 - zs).sum() + self.betas[1]) - \
                   betaln(*self.betas)

        return log_odd_term + log_prob

    def fprime(self, mu, t, s, eps=0):
        raise NotImplementedError

    @confidence_interval
    def construct(self, xs, eps=1e-3, tol=1e-5, verbose=False, batch=False, log_every=100, **kwargs):
        raise NotImplementedError

    def plot(self, delta, xs, upbd=1, every=10, ax=None, legend=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)
        ms = np.arange(0.01, 1, 0.005)

        fs = []
        for t in tqdm(range(1, len(xs) + 1)):
            if t % every == 0:
                fs = np.array([self.f(m, upbd, t, xs[:t]) for m in ms])
                # fs[fs == np.inf] = 1e3
                if 'label' not in kwargs:
                    kwargs['label'] = 'TruncatedKT'
                cummax = np.maximum.accumulate(xs[:t])[-1]
                ax.plot(upbd * ms, fs, **kwargs)
                ax.axhline(np.log(1 / delta), linestyle='--')
                ax.axvline(cummax, linestyle='--', c='red')
                if legend:
                    ax.legend()

        # print(len(xs), self.f(1., len(xs), xs))
        # print(list(zip(ms, fs)))
        return fs


class MultiHorseRaceCI:
    def __init__(self, M=2, betas=None):
        self.M = M
        self.betas = .5 * np.ones((self.M,)) if betas is None else np.array(betas)

    def f(self, qs, ys, eps=0, only_last=False):
        # qs: (n, M)
        # ys: a sequence of M-dim. vectors; (M, T)
        assert qs.shape[-1] == ys.shape[0]
        assert ys.shape[0] == self.M
        mask = ((qs < 0).sum(axis=-1) + (qs > 1).sum(axis=-1)).astype(bool).astype(float)  # (n, )
        mask[mask == 1.] = np.inf

        if only_last:
            ks = ys.sum(axis=-1)  # (M, )
            return np.nan_to_num(
                mask +
                multibetaln(ks + self.betas) +
                - multibetaln(self.betas) +
                - np.einsum('m,nm->n', ks, np.log(qs)),
                nan=1e3, posinf=1e3, neginf=-1e3,
            )  # (n, )
        else:
            csys = ys.cumsum(axis=-1)  # (M, T)
            return np.nan_to_num(
                mask[:, np.newaxis] +  # (n, 1)
                multibetaln(csys + self.betas[:, np.newaxis]) +
                - multibetaln(self.betas) +
                - np.einsum('mt,nm->nt', csys, np.log(qs)),
                nan=1e3, posinf=1e3, neginf=-1e3,
            )  # (n, T)
