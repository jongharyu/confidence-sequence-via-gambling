import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import betaln, logsumexp
from tqdm import tqdm

from methods.scalar.base import confidence_interval, ConfidenceSequence
from methods.scalar.kt import TwoHorseRaceCS
from methods.scalar.lbup import UnboundedLowerBoundUniversalPortfolioCS
from methods.scalar.up import UniversalPortfolioCS


class UnboundedHorseRaceCS(TwoHorseRaceCS):
    def f(self, m, t, xs, eps=0):
        cs = np.maximum.accumulate(xs)
        zs = xs / cs
        log_odd_term = (
            zs * np.nan_to_num(np.log(np.float64(1.0) / (m / cs)), nan=0.0, posinf=0.0)
            + (1 - zs)
            * np.nan_to_num(
                np.log(np.float64(1.0) / (1 - np.minimum(m / cs, np.ones_like(xs)))),
                nan=0.0,
                posinf=0.0,
            )
        ).sum()
        log_prob = betaln(
            zs.sum() + self.betas[0], (1 - zs).sum() + self.betas[1]
        ) - betaln(*self.betas)

        return log_odd_term + log_prob

    def fprime(self, m, t, s, eps=0):
        raise NotImplementedError

    @confidence_interval
    def construct(
        self,
        xs,
        eps=1e-3,
        tol=1e-5,
        verbose=False,
        batch=False,
        log_every=100,
        **kwargs,
    ):
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
                if "label" not in kwargs:
                    kwargs["label"] = "UnbddKT"
                cummax = np.maximum.accumulate(xs[:t])[-1]
                ax.plot(ms, fs, **kwargs)
                ax.axhline(np.log(1 / delta), linestyle="--")
                ax.axvline(cummax, linestyle="--", c="red")
                if legend:
                    ax.legend()

        # print(len(xs), self.f(1., len(xs), xs))
        # print(list(zip(ms, fs)))
        return fs


class UnboundedUniversalPortfolioCS(UniversalPortfolioCS):
    def __init__(self, betas=(1 / 2, 1 / 2), flip=False):
        super().__init__()
        self.betas = betas
        self.flip = flip

    def f(self, m, t, logweights, eps=0, verbose=False):
        # log(wealth of UP)
        if verbose:
            print("t, m:", t, m)
        if self.flip:
            m = 1 - m
        return logsumexp(logweights - np.arange(t + 1) * np.log(m + eps))

    def fprime(self, m, t, logweights, eps=0):
        if self.flip:
            m = 1 - m
        # derivative
        base = logweights - np.arange(t + 1) * np.log(m + eps)
        log_denom = logsumexp(base)  # = self.f(m, t, logweights, eps, verbose=False)
        return -np.exp(
            logsumexp(base[1:] + np.log(np.arange(1, t + 1)) - np.log(m + eps))
            - log_denom
        )

    # def update_logsumprod(self, logsumprod, x, eps=1e-5):
    #     logsumprod = logsumexp([np.pad(logsumprod + np.log(x + eps), (1, 0), constant_values=(-np.inf)),
    #                             np.pad(logsumprod, (0, 1), constant_values=(-np.inf))],
    #                            axis=0)
    #     return logsumprod

    def update_logsumprod(self, logsumprod, x, eps=1e-5):
        padded_shape = (len(logsumprod) + 1,)
        neg_inf_array = -np.inf * np.ones(padded_shape)

        # Precompute log(x + eps) once
        log_x_eps = np.log(x + eps)

        # Directly assign values to avoid padding
        arr1 = neg_inf_array.copy()
        arr1[1:] = logsumprod + log_x_eps
        arr2 = neg_inf_array.copy()
        arr2[:-1] = logsumprod

        logsumprod = logsumexp([arr1, arr2], axis=0)

        return logsumprod

    def compute_logweights(self, t, logsumprod):
        return logsumprod + (
            betaln(
                np.arange(t + 1) + self.betas[0], t - np.arange(t + 1) + self.betas[1]
            )
            - betaln(*self.betas)
        )

    @confidence_interval
    def construct(
        self,
        delta,
        xs,
        eps=0,
        tol=1e-5,
        verbose=False,
        log_every=100,
        tqdm_=True,
        **kwargs,
    ):
        tqdm_ = tqdm if tqdm_ else lambda x: x
        lower_ci = np.zeros_like(xs).astype(float)
        upper_ci = np.ones_like(xs).astype(float)

        logsumprod = np.array([0.0])
        logweights = np.array([0.0])

        xinit_low = 0.01
        xinit_up = 0.99

        telapsed = []
        start = time.time()
        for t in tqdm_(range(1, len(xs) + 1)):
            x = xs[t - 1]
            logsumprod = self.update_logsumprod(logsumprod, x)
            logweights = self.compute_logweights(t, logsumprod)

            if verbose:
                # to see if log wealth(mu_hat) <= 0 always:
                mu_hat = xs[:t].mean()
                f_mu_hat = self.f(mu_hat, t, logweights)
                if f_mu_hat >= 0:
                    print("t={}, mu_hat={}, f_t(mu_hat)={}".format(t, mu_hat, f_mu_hat))
                    print(
                        "t={}, mu_hat={}, f_t'(mu_hat)={}".format(
                            t, mu_hat, self.fprime(mu_hat, t, logweights)
                        )
                    )

            lower_ci[t - 1] = self.find_root(
                delta,
                t,
                logweights,
                xinit=xinit_low,
                xmin=0,
                xmax=1,
                tol=tol,
                verbose=verbose,
            )
            xinit_low = lower_ci[t - 1] if not np.isnan(lower_ci[t - 1]) else 1e-6

            if t % log_every == 0:
                end = time.time()
                telapsed.append(end - start)
                # print(t, end=' ')
                start = end

        return lower_ci, upper_ci, telapsed, logweights


class UnboundedHybridUniversalPortfolioCS(ConfidenceSequence):
    def __init__(self, n=1, tup=50, betas=(1 / 2, 1 / 2)):
        super().__init__()
        self.n = n  # the approximation order in LBUP
        self.tup = tup  # how long you will run UP at the beginning
        self.betas = betas  # UP parameter

    @confidence_interval
    def construct(
        self, delta, xs, eps=0, tol=1e-5, verbose=False, log_every=100, **kwargs
    ):
        lower_ci = np.zeros_like(xs).astype(float)
        upper_ci = np.zeros_like(xs).astype(float)

        # Run UnboundedUP up until self.tup round
        lower_ci[: self.tup], _, telapsed_up, logweights = (
            UnboundedUniversalPortfolioCS(
                betas=self.betas
            ).construct(
                delta,
                xs[: self.tup],
                eps=eps,
                tol=tol,
                verbose=verbose,
                log_every=log_every,
                do_not_apply_wor=True,
                **kwargs,
            )
        )

        # compute cumulative sums till t=self.tup which are to be used in the prior for LBUP
        sums0 = np.stack([(xs[: self.tup] ** k) for k in range(2 * self.n + 1)]).sum(
            axis=1
        )

        # Run LBUP from then
        lower_ci[self.tup :], _, telapsed_lbup = (
            UnboundedLowerBoundUniversalPortfolioCS(
                self.n, sums0=sums0, tup=self.tup, logweights=logweights
            ).construct(
                delta,
                xs[self.tup :],
                eps=eps,
                tol=tol,
                verbose=verbose,
                log_every=log_every,
                do_not_apply_wor=True,
                **kwargs,
            )
        )

        return lower_ci, upper_ci, np.array(telapsed_up + telapsed_lbup)

    def plot(self, delta, xs, every=10, ax=None, legend=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)
        ms = np.arange(0.01, 1, 0.01)

        # Run UP up until self.tup round
        *_, logweights = UnboundedUniversalPortfolioCS(betas=self.betas).plot(
            delta, xs[: self.tup], every, ax, legend, **kwargs
        )

        # compute cumulative sums till t=self.tup which are to be used in the prior for LBUP
        sums0 = np.stack([(xs[: self.tup] ** k) for k in range(2 * self.n + 1)]).sum(
            axis=1
        )

        lbup = UnboundedLowerBoundUniversalPortfolioCS(
            self.n, sums0=sums0, tup=self.tup, logweights=logweights
        )

        sums = (
            np.stack([(xs[self.tup :] ** k) for k in range(2 * self.n + 1)])
            .cumsum(axis=1)
            .T
        )  # (T, 2 * n + 1)

        # Run LBUP from then
        fs = []
        for t in tqdm(range(self.tup + 1, len(xs) + 1)):
            if t % every == 0:
                mu_hat = (sums[t - self.tup - 1, 1] + sums0[1]) / (
                    sums[t - self.tup - 1, 0] + sums0[0]
                )
                print(
                    "t={}, f(mu_hat)={}".format(
                        t + self.tup, lbup.f(mu_hat, sums[t - self.tup - 1])
                    )
                )
                fs = np.zeros_like(ms)
                for i, m in enumerate(ms):
                    fs[i] = lbup.f(m, sums[t - self.tup - 1])
                if "label" not in kwargs:
                    kwargs["label"] = "UnboundedHybridUP"
                kwargs["label"] += f" (order={self.n}; t={t})"
                ax.plot(ms, fs, **kwargs)
                ax.axhline(np.log(1 / delta), linestyle="--")
                ax.axvline(x=mu_hat)
                if legend:
                    ax.legend()

        return fs
