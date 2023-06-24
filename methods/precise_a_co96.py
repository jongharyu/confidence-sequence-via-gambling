import time

import numpy as np
from scipy.special import gammaln

from methods.base import ConfidenceSequence
from utils import confidence_interval


def max_logwealth_fan3_lcb(m, mu_hat, var_hat, t):
    if m == 0.0:
        max_logwealth = (0.5 * (mu_hat ** 2) / (var_hat + mu_hat ** 2)) * t
    elif m == mu_hat:
        max_logwealth = 0.0
    else:
        A = (mu_hat - m) / m
        B = (var_hat + (mu_hat - m) ** 2) / m ** 2
        lam = A / (A + B)
        max_logwealth = (A * A / (A + B) - (-np.log(1 - lam) - lam) * B) * t
    return max_logwealth

def max_logwealth_fan3_ucb(m, mu_hat, var_hat, t):
    if m == 1.0:
        max_logwealth = (0.5 * ((1 - mu_hat) ** 2) / (var_hat + (1 - mu_hat) ** 2)) * t
    elif m == mu_hat:
        max_logwealth = 0.0
    else:
        A = (m - mu_hat) / (1 - m)
        B = (var_hat + (m - mu_hat) ** 2) / (1 - m) ** 2
        lam = A / (A + B)
        max_logwealth = (A * A / (A + B) - (-np.log(1 - lam) - lam) * B) * t
    return max_logwealth


def kl(p, q):
    if p == 0:
        val = (1 - p) * np.log((1 - p) / (1 - q))
    elif p == 1:
        val = p * np.log(p / q)
    else:
        val = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return val


def max_logwealth_kl(m, mu_hat, var_hat, t):
    max_logwealth = t * kl(mu_hat, m)
    return max_logwealth


def bsearch(fn, lb, ub, tol=1e-6, eps=1e-5):
    fnlb = fn(lb + eps)
    fnub = fn(ub - eps)
    assert fnlb * fnub != 0.0
    # assert (fnlb <= 0 and fnub >= 0) or (fnlb >= 0 and fnub <= 0), (lb, fnlb, ub, fnub)
    sign_fnlb = 1.0
    if fnlb <= 0 and fnub >= 0:
        sign_fnlb = -1.0
    max_iter = 1000
    i_iter = 0
    while ub - lb >= tol and i_iter <= max_iter:
        mid = (lb + ub) / 2
        val = fn(mid)
        if val * sign_fnlb > 0.0:
            lb = mid
        else:
            ub = mid
        i_iter += 1
    if i_iter >= max_iter:
        print("WARNING: max_iter has reached")
    return lb, ub


class PRECiSE_A_CO96(ConfidenceSequence):
    """
    PRECISE_A_CO96    Portfolio REgret for Confidence SEquences with Approximation using Cover and Ordentlich [1996].
    [L,U] = PRECISE_A_CO96(X,delta) produces two matrices, of the same
    dimension as X and with lower and upper confidence sequences with
    probability of error delta. X is numer-of-samples by number-of-repetitions.

    This algorithm is described in  Orabona and Jun, "Tight Concentrations
    and Confidence Sequences from the Regret of Universal Portfolio", ArXiv 2021.
    """
    def __init__(self):
        super().__init__()

    @confidence_interval
    def construct(self, delta, xs, eps=1e-7, verbose=False, log_every=100, **kwargs):
        reg = lambda t: np.log(np.sqrt(np.pi)) + gammaln(t + 1) - gammaln(t + 0.5)

        n_algo = 2
        rdata = np.zeros((len(xs), n_algo, 2))
        runningmax = np.zeros(n_algo)
        runningmin = np.ones(n_algo)

        telapsed = []
        start = time.time()
        for t in range(len(xs)):
            data = xs[:t + 1]
            me = np.mean(data)
            va = np.var(data, ddof=1) if t > 0 else 0.

            lcb = np.zeros(n_algo)
            ucb = np.ones(n_algo)
            i_algo = 0

            # --- fan
            i_algo += 1
            rhs = reg(t + 1) + np.log(1 / delta)

            lb = 0.0
            ub = me
            lcbmaxfn = lambda m: max(max_logwealth_fan3_lcb(m, me, va, t + 1), max_logwealth_kl(m, me, va, t + 1))
            # print("DEBUG(LCB):", me, va, t, lcbmaxfn(0.5))
            if lb == ub or lcbmaxfn(lb) - rhs <= 0:
                lcb[i_algo - 1] = 0.0
            else:
                lcb[i_algo - 1], _ = bsearch(lambda m: lcbmaxfn(m) - rhs, lb, ub, eps)

            lb = me
            ub = 1.0
            ucbmaxfn = lambda m: max(max_logwealth_fan3_ucb(m, me, va, t + 1), max_logwealth_kl(m, me, va, t + 1))
            # print("DEBUG(UCB):", me, va, t, ucbmaxfn(0.5))
            if lb == ub or ucbmaxfn(ub) - rhs <= 0:
                ucb[i_algo - 1] = 1.0
            else:
                _, ucb[i_algo - 1] = bsearch(lambda m: ucbmaxfn(m) - rhs, lb, ub, eps)

            runningmax[:] = np.maximum(runningmax, lcb)
            runningmin[:] = np.minimum(runningmin, ucb)
            rdata[t, :, 0] = runningmax
            rdata[t, :, 1] = runningmin

            if t % log_every == 0:
                end = time.time()
                telapsed.append(end - start)
                if verbose:
                    print(t, end=' ')
                start = end

        lower_ci = rdata[:, 0, 0]
        upper_ci = rdata[:, 0, 1]

        return lower_ci, upper_ci, telapsed
