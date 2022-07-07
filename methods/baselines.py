import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect, fsolve
from scipy.special import betaln, digamma

from utils import binary_entropy, confidence_interval


class ConfidenceSeqeunce:
    def __init__(self, delta):
        self.delta = delta

    def f(self, x, *args):
        raise NotImplementedError

    def fprime(self, x, *args):
        raise NotImplementedError

    @staticmethod
    def empirical_mean(xs):
        ts = np.arange(1, len(xs) + 1)
        ss = xs.cumsum()
        mu_hats = ss / ts
        return mu_hats

    # root finding
    def find_root(self, *args,
                  xmin=-np.inf, xmax=np.inf, xinit=-1,
                  maxiter=100, tol=1e-5, verbose=False):
        assert np.all(xmin > -np.inf) and np.all(xmax < np.inf)

        def f(x):
            return self.f(x, *args) - np.log(1 / self.delta)

        # return the positive root
        if xinit != -1:
            x = xinit
        else:
            x = (xmax - xmin) / 2  # arbitrary initialization
        cnt = 0
        while 1:
            xprev = x
            x = xprev - (f(xprev)) / self.fprime(xprev, *args)  # Newton--Raphson update
            x = np.minimum(xmax, np.maximum(xmin, x))
            if verbose:
                print('xprev, x:', xprev, x)
            cnt += 1
            if (np.abs(x - xprev) < tol).all() or cnt > maxiter:
                break

        if verbose:
            print('cnt, diff:', cnt, np.abs(x - xprev))
        return x

    def find_root_fsolve(self, *args, xinit):
        def f(x):
            return self.f(x, *args) - np.log(1 / self.delta)

        return fsolve(f, x0=xinit)

    def find_root_bisect(self, *args,
                         xinits,
                         tol=1e-5,
                         maxiter=16, verbose=False):
        def f(x):
            return self.f(x, *args) - np.log(1 / self.delta)

        # if using scipy.optimize.bisect
        # try:
        #     return bisect(f, *xinits, xtol=tol)
        # except ValueError:
        #     return -1

        # optional args: maxiter=16, verbose=False
        def compare_signs(f1, f2):
            return np.sign(f1) != np.sign(f2)

        xlow, xhi = xinits
        assert xlow < xhi, (xlow, xhi)

        flow, fhi = f(xlow), f(xhi)
        flow = np.inf if np.isnan(flow) else flow
        fhi = np.inf if np.isnan(fhi) else fhi

        if not compare_signs(flow, fhi):
            if flow > 0 and fhi > 0:
                if verbose:
                    print("Wealth not above the threshold!", (xlow, xhi), (flow, fhi))
            if flow < 0 and fhi < 0:
                return np.nan
            if xlow < 1e-5:
                return 0
            elif xhi > 1 - 1e-5:
                return 1

        cnt = 0
        while 1:
            xmid = (xlow + xhi) / 2
            fmid = f(xmid)
            if np.isnan(fmid):
                fmid = np.inf
            if verbose:
                print("(xlow, xhi)", (xlow, xhi), "(flow, fmid, fhi)", (flow, fmid, fhi))

            if compare_signs(flow, fmid):
                xhi, fhi = xmid, fmid
            elif compare_signs(fmid, fhi):
                xlow, flow = xmid, fmid
            else:
                print("xinits", xinits, "(xlow, xhi)", (xlow, xhi), "(flow, fmid, fhi)", (flow, fmid, fhi))
                raise ValueError

            cnt += 1
            if cnt > maxiter or np.abs(xhi - xlow) < tol:
                break

        if verbose:
            print('cnt, diff:', cnt, np.abs(xhi - xlow))

        root = xlow if flow > 0 else xhi
        assert xinits[0] <= root <= xinits[1]
        if verbose:
            # for debugging
            print("\t=>", xinits, (xlow, flow), (xhi, fhi), root)

        return root


# based on discrete-coin betting + "naive embedding"
class CoinBettingCI(ConfidenceSeqeunce):
    def __init__(self, delta):
        super().__init__(delta)

    def f(self, x, t):
        return t * np.log(2) + betaln((t + x + 1) / 2, (t - x + 1) / 2) - betaln(1 / 2, 1 / 2)

    def fprime(self, x, t):
        return 1 / 2 * (digamma((t + x + 1) / 2) - digamma((t - x + 1) / 2))

    @confidence_interval
    def construct(self, xs, verbose=False, **kwargs):
        ts = np.arange(1, len(xs) + 1)
        mu_hats = xs.cumsum() / ts
        roots = self.find_root(ts, xmin=-ts, xmax=ts, verbose=verbose)

        lower_ci = mu_hats - roots / ts
        upper_ci = mu_hats + roots / ts

        return lower_ci, upper_ci

    def plot(self, xs, every=10, ax=None, legend=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)

        raise NotImplementedError


# based on discrete-coin betting + "tighter embedding"
class HorseRaceCI(ConfidenceSeqeunce):
    def __init__(self, delta, betas=(1 / 2, 1 / 2)):
        super().__init__(delta)
        self.betas = betas

    def f(self, mu, t, s, eps=0):
        # negative log bernoulli probability with count s at time step t
        return - s * np.log(mu + eps) - (t - s) * np.log(1 - mu + eps) \
               + betaln(s + self.betas[0], t - s + self.betas[1]) - betaln(*self.betas)

    def fprime(self, mu, t, s, eps=0):
        # derivative
        return - s / (mu + eps) + (t - s) / (1 - mu + eps)

    @confidence_interval
    def construct(self, xs, eps=1e-3, tol=1e-5, verbose=False, batch=False, log_every=100, **kwargs):
        ts = np.arange(1, len(xs) + 1)
        ss = xs.cumsum()
        telapsed = []

        if batch:
            lower_ci = self.find_root(ts, ss, xinit=eps, xmin=0, xmax=1, verbose=verbose)
            upper_ci = self.find_root(ts, ss, xinit=1 - eps, xmin=0, xmax=1, verbose=verbose)
        else:
            lower_ci = np.zeros_like(xs).astype(float)
            upper_ci = np.ones_like(xs).astype(float)

            start = time.time()
            for t in range(1, len(xs) + 1):
                mu_hat = ss[t - 1] / t
                if mu_hat == 0:
                    mu_hat = eps
                if mu_hat == 1:
                    mu_hat = 1 - eps

                # use scipy's bisect with a customized initialization rule
                xinit_low = lower_ci[t - 2] if t > 1 else eps
                if self.f(xinit_low, t, ss[t - 1]) < np.log(1 / self.delta):
                    lower_ci[t - 1] = lower_ci[t - 2]
                else:
                    lower_ci[t - 1] = self.find_root_bisect(t, ss[t - 1],
                                                            xinits=(lower_ci[t - 2], mu_hat),
                                                            tol=tol,
                                                            verbose=verbose)
                    if lower_ci[t - 1] == -1:
                        print("bisect encounters ValueError!")
                        lower_ci[t - 1] = lower_ci[t - 2]

                xinit_up = upper_ci[t - 2] if t > 1 else 1 - eps
                if self.f(xinit_up, t, ss[t - 1]) < np.log(1 / self.delta):
                    upper_ci[t - 1] = upper_ci[t - 2]
                else:
                    upper_ci[t - 1] = self.find_root_bisect(t, ss[t - 1],
                                                            xinits=(mu_hat, upper_ci[t - 2]),
                                                            tol=tol,
                                                            verbose=verbose)
                    if upper_ci[t - 1] == -1:
                        print("bisect encounters ValueError!")
                        upper_ci[t - 1] = upper_ci[t - 2]

                if t % log_every == 0:
                    end = time.time()
                    telapsed.append(end - start)
                    print(t, end=' ')
                    start = end

        return lower_ci, upper_ci, telapsed

    @confidence_interval
    def construct_outer(self, xs):
        ts = np.arange(1, len(xs) + 1)
        ss = xs.cumsum()

        mu_hats = ss / ts
        logqkt = betaln(ss + self.betas[0], ts - ss + self.betas[1]) - betaln(*self.betas)
        gs = 1 / ts * (np.log(1 / self.delta) - logqkt) - binary_entropy(mu_hats)

        lower_ci = mu_hats - np.sqrt(gs / 2)
        upper_ci = mu_hats + np.sqrt(gs / 2)

        return lower_ci, upper_ci

    def plot(self, xs, every=10, ax=None, legend=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)
        mus = np.arange(0.01, 1, 0.01)

        fs = []
        for t in range(1, len(xs) + 1):
            if t % every == 0:
                fs = self.f(mus, t, xs[:t].sum())
                if 'label' not in kwargs:
                    kwargs['label'] = 'HR'
                ax.plot(mus, fs, **kwargs)
                ax.axhline(np.log(1 / self.delta), linestyle='--')
                if legend:
                    ax.legend()

        return fs
