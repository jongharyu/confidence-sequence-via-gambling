import time

import numpy as np
import scipy
import scipy.integrate as integrate
from matplotlib import pyplot as plt
from scipy.special import binom, logsumexp, gammaln, gammainc
from tqdm import tqdm

import methods.lbup_integrand as lbup_integrand
from methods.scalar.base import ConfidenceSequence, confidence_interval
from methods.scalar.up import UniversalPortfolioCS


def logbinom(n, k):
    return np.log(binom(n, k))


def stable_expsub(a, b):
    # Convert inputs to numpy arrays if they are not already
    a = np.asarray(a)
    b = np.asarray(b)

    # Create an output array to handle the cases
    result = np.zeros_like(a)

    # Case when a == b
    mask_equal = a == b
    result[mask_equal] = 0.0

    # Case when a > b
    mask_greater = a > b
    result[mask_greater] = np.exp(b[mask_greater]) * np.expm1(
        a[mask_greater] - b[mask_greater]
    )

    # Case when a < b
    mask_less = a < b
    result[mask_less] = -np.exp(a[mask_less]) * np.expm1(b[mask_less] - a[mask_less])

    return result


class TruncatedGamma:
    def __init__(self, rhos, eta, eps=0, use_cython=True):
        self.rhos = np.atleast_1d(rhos)
        self.eta = eta
        self.eps = np.max(eps, 0)
        self.use_cython = use_cython

        assert len(self.rhos) % 2 == 1

    @property
    def n(self):
        return (len(self.rhos) + 1) // 2

    def log_phi(self, x):
        assert 0 <= x <= 1
        return np.sum(
            [self.rhos[k - 1] * ((1 - x) ** k) / k for k in range(1, 2 * self.n)]
        ) + self.eta * np.log(x)

    @property
    def log_z(self):
        if self.n == 1:
            # log{Z(rho, eta)}
            if self.rhos[0] > 0:
                return (
                    self.rhos[0]
                    - (self.eta + 1) * np.log(self.rhos[0] + self.eps)
                    + gammaln(self.eta + 1)
                    + np.log(gammainc(self.eta + 1, self.rhos[0] + self.eps))
                )
            elif self.rhos[0] == 0:
                return -np.log(self.eta + 1)
            else:
                return np.log(
                    integrate.quad(lambda x: np.exp(self.log_phi(x)), 0, 1)[0]
                )
        else:
            # log{Z(rhos, eta)}
            if self.use_cython:
                # Turn the Cython C function into a LowLevelCallable
                # using Cython seems ~33% faster!
                phi = scipy.LowLevelCallable.from_cython(lbup_integrand, "phi")
                args = (self.n, *self.rhos, self.eta)
            else:
                phi = lambda x: np.exp(self.log_phi(x))
                args = ()

            z, err = integrate.quad(phi, 0, 1, args=args)
            return np.log(z)


class TruncatedGammaParams:
    # Define rho1 and eta1 with Y^t
    # rho2 and eta2 are defined symmetrically by replacing {Y^t, m} with {(1-Y^t), (1-m)}
    def __init__(self, n, use_cython=True):
        self.n = n
        self.use_cython = use_cython

    @staticmethod
    def g(m, k, sums, eps=1e-5):
        log_even = logsumexp(
            np.stack(
                [
                    logbinom(k, j) + np.log(sums[j]) - j * np.log(m + eps)
                    for j in range(0, k + 1, 2)
                ],
                axis=0,
            ),
            axis=0,
        )
        log_odd = logsumexp(
            np.stack(
                [
                    logbinom(k, j) + np.log(sums[j]) - j * np.log(m + eps)
                    for j in range(1, k + 1, 2)
                ],
                axis=0,
            ),
            axis=0,
        )
        print(f"Debugging g: m={m}, log_even={log_even}, log_odd={log_odd}")
        return stable_expsub(log_even, log_odd)  # np.exp(log_even) - np.exp(log_odd)

    #         return np.sum([binom(k, j) * sums[j] / ((-m) ** j) for j in range(k + 1)])

    def compute_params(self, m, sums):
        assert len(sums) == 2 * self.n + 1, (len(sums), self.n)
        eta = self.g(m, 2 * self.n, sums)
        rhos = eta - np.array([self.g(m, k, sums) for k in range(1, 2 * self.n)])
        return rhos, eta

    def compute_log_z(self, m, sums):
        rhos, eta = self.compute_params(m, sums)
        print(
            f"Debugging TruncatedGammaParams1 m={m}, sums={sums}, rhos={rhos}, eta={eta}"
        )
        if isinstance(m, float):
            base_log_z = TruncatedGamma(rhos, eta, use_cython=self.use_cython).log_z
        else:
            base_log_z = np.array(
                [
                    TruncatedGamma(rhos[:, i], eta[i], use_cython=self.use_cython).log_z
                    for i in range(len(m))
                ]
            )
        print(f"Debugging TruncatedGammaParams2 base_log_z={base_log_z}")
        return np.log(1 - m) + base_log_z


class StitchedTruncatedGamma:
    def __init__(self, m, rhos1, eta1, rhos2, eta2, use_cython=True):
        self.tg1 = TruncatedGamma(rhos1, eta1, use_cython=use_cython)
        self.tg2 = TruncatedGamma(rhos2, eta2, use_cython=use_cython)
        self.m = m

    def log_phi(self, b):
        assert 0 <= b <= 1
        if 0 <= b <= self.m:
            return self.tg2.log_phi(b / self.m)
        else:
            return self.tg1.log_phi((1 - b) / (1 - self.m))

    @property
    def log_z(self):
        log_z2 = self.tg2.log_z
        log_z1 = self.tg1.log_z
        # print((np.log(self.m), log_z2), (np.log(1 - self.m), log_z1))
        return logsumexp([log_z2 + np.log(self.m), log_z1 + np.log(1 - self.m)])


class StitchedTruncatedGammaParams:
    def __init__(self, n):
        self.tg_params = TruncatedGammaParams(n)

    def compute_log_z(self, m, sums, sums_c):
        log_z1 = self.tg_params.compute_log_z(m, sums)
        log_z2 = self.tg_params.compute_log_z(1 - m, sums_c)
        if log_z1 == -np.inf:
            return log_z2
        elif log_z2 == -np.inf:
            return log_z1
        else:
            return logsumexp([log_z1, log_z2])


class LowerBoundUniversalPortfolioCS(ConfidenceSequence):
    def __init__(
        self,
        n,
        sums0=0,
        sums_c0=0,
        tup=0,
        betas=(1 / 2, 1 / 2),
        logweights=None,
        use_cython=True,
    ):
        super().__init__()
        self.n = n
        self.use_cython = use_cython
        self.tg_params = TruncatedGammaParams(self.n, self.use_cython)

        # for rhos and eta for a prior
        self.sums0 = sums0 if not isinstance(sums0, int) else np.zeros(2 * self.n + 1)
        self.sums_c0 = (
            sums_c0 if not isinstance(sums_c0, int) else np.zeros(2 * self.n + 1)
        )

        # for piggybacking UP (these are used in HybridUP)
        self.tup = tup
        self.betas = betas
        self.logweights = logweights

    def f(self, m, sums, sums_c, verbose=False):
        # f(m, st, sst) = log((Z1t + Z2t) / (Z10 + Z20))
        log_numer = logsumexp(
            [
                self.tg_params.compute_log_z(m, sums + self.sums0),
                self.tg_params.compute_log_z(1 - m, sums_c + self.sums_c0),
            ]
        )
        log_denom = logsumexp(
            [
                self.tg_params.compute_log_z(m, self.sums0),
                self.tg_params.compute_log_z(1 - m, self.sums_c0),
            ]
        )
        val = log_numer - log_denom

        if self.logweights is not None:
            val += UniversalPortfolioCS(betas=self.betas).f(
                m, self.tup, self.logweights
            )

        return np.nan_to_num(val, nan=np.inf)

    def fprime(self, x, *args):
        raise NotImplementedError

    @confidence_interval
    def construct(
        self, delta, xs, tol=1e-5, eps=1e-5, verbose=False, log_every=100, **kwargs
    ):
        # Note: if eps is too small, then due to numerical instability of the definite integrals,
        #       the behavior may be erratic
        lower_ci = np.zeros_like(xs).astype(float)
        upper_ci = np.ones_like(xs).astype(float)

        sums = (
            np.stack([(xs**k) for k in range(2 * self.n + 1)]).cumsum(axis=1).T
        )  # (T, 2 * n + 1)
        sums_c = (
            np.stack([((1 - xs) ** k) for k in range(2 * self.n + 1)]).cumsum(axis=1).T
        )  # (T, 2 * n + 1)

        telapsed = []
        start = time.time()
        for t in tqdm(range(1, len(xs) + 1)):
            mu_hat = (sums[t - 1, 1] + self.sums0[1]) / (sums[t - 1, 0] + self.sums0[0])

            # use scipy's fsolve (somehow doesn't work properly)
            # lower_ci[t - 1] = self.find_root_fsolve(
            #     sums[t - 1],
            #     sums_c[t - 1],
            #     xinit=(lower_ci[t - 2] + mu_hat) / 2
            # )
            # upper_ci[t - 1] = self.find_root_fsolve(
            #     sums[t - 1],
            #     sums_c[t - 1],
            #     xinit=(upper_ci[t - 2] + mu_hat) / 2
            # )

            # use scipy's bisect with a customized initialization rule
            xinit_low = lower_ci[t - 2] if t > 1 else eps
            if self.f(xinit_low, sums[t - 1], sums_c[t - 1]) < np.log(1 / delta):
                lower_ci[t - 1] = xinit_low
            else:
                lower_ci[t - 1] = self.find_root_bisect(
                    delta,
                    sums[t - 1],
                    sums_c[t - 1],
                    xinits=(xinit_low, mu_hat),
                    tol=tol,
                    verbose=verbose,
                )
                if lower_ci[t - 1] == -1 or np.isnan(lower_ci[t - 1]):
                    print("bisect encounters ValueError!")
                    lower_ci[t - 1] = lower_ci[t - 2]

            xinit_up = upper_ci[t - 2] if t > 1 else 1 - eps
            if self.f(xinit_up, sums[t - 1], sums_c[t - 1]) < np.log(1 / delta):
                upper_ci[t - 1] = xinit_up
            else:
                upper_ci[t - 1] = self.find_root_bisect(
                    delta,
                    sums[t - 1],
                    sums_c[t - 1],
                    xinits=(mu_hat, xinit_up),
                    tol=tol,
                    verbose=verbose,
                )
                if upper_ci[t - 1] == -1 or np.isnan(upper_ci[t - 1]):
                    print("bisect encounters ValueError!")
                    upper_ci[t - 1] = upper_ci[t - 2]

            if (t + self.tup) % log_every == 0:
                end = time.time()
                telapsed.append(end - start)
                # print(t + self.tup, end=' ')
                start = end

        return lower_ci, upper_ci, telapsed

    def plot(self, delta, xs, every=10, ax=None, legend=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)
        ms = np.arange(0.01, 1, 0.01)

        sums = (
            np.stack([(xs**k) for k in range(2 * self.n + 1)]).cumsum(axis=1).T
        )  # (T, 2 * n + 1)
        sums_c = (
            np.stack([((1 - xs) ** k) for k in range(2 * self.n + 1)]).cumsum(axis=1).T
        )  # (T, 2 * n + 1)

        fs = []
        for t in range(1, len(xs) + 1):
            if (t + self.tup) % every == 0:
                print(t + self.tup)
                fs = np.zeros_like(ms)
                for i, m in enumerate(ms):
                    fs[i] = self.f(m, sums[t - 1], sums_c[t - 1])
                if "label" not in kwargs:
                    kwargs["label"] = "LBUP"
                kwargs["label"] += f" (n={self.n})"
                ax.plot(ms, fs, **kwargs)
                ax.axhline(np.log(1 / delta), linestyle="--")
                if legend:
                    ax.legend()

        return fs


class HybridUniversalPortfolioCS(ConfidenceSequence):
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

        # Run UP up until self.tup round
        lower_ci[: self.tup], upper_ci[: self.tup], telapsed_up, logweights = (
            UniversalPortfolioCS(betas=self.betas).construct(
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
        sums_c0 = np.stack(
            [((1 - xs[: self.tup]) ** k) for k in range(2 * self.n + 1)]
        ).sum(axis=1)

        # Run LBUP from then
        lower_ci[self.tup :], upper_ci[self.tup :], telapsed_lbup = (
            LowerBoundUniversalPortfolioCS(
                self.n,
                sums0=sums0,
                sums_c0=sums_c0,
                tup=self.tup,
                logweights=logweights,
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
        *_, logweights = UniversalPortfolioCS(betas=self.betas).plot(
            delta, xs[: self.tup], every, ax, legend, **kwargs
        )

        # compute cumulative sums till t=self.tup which are to be used in the prior for LBUP
        sums0 = np.stack([(xs[: self.tup] ** k) for k in range(2 * self.n + 1)]).sum(
            axis=1
        )
        sums_c0 = np.stack(
            [((1 - xs[: self.tup]) ** k) for k in range(2 * self.n + 1)]
        ).sum(axis=1)

        lbup = LowerBoundUniversalPortfolioCS(
            self.n, sums0=sums0, sums_c0=sums_c0, tup=self.tup, logweights=logweights
        )

        sums = (
            np.stack([(xs[self.tup :] ** k) for k in range(2 * self.n + 1)])
            .cumsum(axis=1)
            .T
        )  # (T, 2 * n + 1)
        sums_c = (
            np.stack([((1 - xs[self.tup :]) ** k) for k in range(2 * self.n + 1)])
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
                        t + self.tup,
                        lbup.f(
                            mu_hat, sums[t - self.tup - 1], sums_c[t - self.tup - 1]
                        ),
                    )
                )
                fs = np.zeros_like(ms)
                for i, m in enumerate(ms):
                    fs[i] = lbup.f(m, sums[t - self.tup - 1], sums_c[t - self.tup - 1])
                if "label" not in kwargs:
                    kwargs["label"] = "HybridUP"
                kwargs["label"] += f" (order={self.n}; t={t})"
                ax.plot(ms, fs, **kwargs)
                ax.axhline(np.log(1 / delta), linestyle="--")
                ax.axvline(x=mu_hat)
                if legend:
                    ax.legend()

        return fs
