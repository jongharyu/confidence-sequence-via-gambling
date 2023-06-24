import time

import numpy as np
from scipy.special import gammaln

from methods.base import ConfidenceSequence
from utils import confidence_interval


def newton_1d_bnd(funfcn, df, df2, ax, bx):
    deriv = np.inf
    x = 0
    x_old = np.inf

    while np.abs(deriv) > 0.001 and np.abs(x - x_old) > 1e-3:
        x_old = x
        deriv = df(x)
        if x == ax and deriv < 0:
            break
        elif x == bx and deriv > 0:
            break
        if np.abs(deriv) > 1e3:
            update = 0.01 * np.sign(deriv)
        else:
            deriv2 = df2(x)
            update = -deriv / deriv2

        x += update
        x = max(min(x, bx), ax)

    fval = funfcn(x)

    return x, fval


def find_max_log_wealth_constrained(g, bmin, bmax):
    myf = lambda b: np.sum(np.log(1 + g * b))
    df = lambda b: np.sum(g / (1 + g * b))
    df2 = lambda b: -np.sum((g / (1 + g * b)) ** 2)
    b_star, fval = newton_1d_bnd(myf, df, df2, max(bmin, -1e10), min(bmax, 1e10))

    return b_star, fval


class PRECiSE_CO96(ConfidenceSequence):
    """
    PRECISE_CO96    Portfolio REgret for Confidence SEquences using Cover and Ordentlich [1996].
    [L,U] = PRECISE_CO96(X,delta) produces two matrices, of the same
    dimension as X and with lower and upper confidence sequences with
    probability of error delta. X is numer-of-samples by number-of-repetitions.

    This algorithm is described in  Orabona and Jun, "Tight Concentrations
    and Confidence Sequences from the Regret of Universal Portfolio", ArXiv 2021.
    """
    def __init__(self, refine=True):
        super().__init__()
        self.refine = refine

    @confidence_interval
    def construct(self, delta, xs, eps=1e-7, verbose=False, log_every=100, **kwargs):
        def func(b, k, t):
            return (k * np.log(b + eps) +
                    (t - k) * np.log(1 - b + eps) +
                    gammaln(t + 1) +
                    2 * gammaln(1 / 2) -
                    gammaln(k + 1 / 2) -
                    gammaln(t - k + 1 / 2))

        lower_ci = np.zeros_like(xs).astype(float)
        upper_ci = np.ones_like(xs).astype(float)

        m_lb_old = eps
        m_ub_old = 1 - eps

        telapsed = []
        start = time.time()
        for t in range(len(xs)):
            if t % 100 == 0:
                print(t, end=' ')

            mu_hat = xs[:t].mean()

            # Upper CI
            m_ub = m_ub_old
            m_lb = max(m_lb_old, mu_hat)

            # compute regret
            m_try = m_ub
            bmax = 1 / m_try
            bmin = -1 / (1 - m_try)

            b_star, log_W_star = find_max_log_wealth_constrained(xs[:t] - m_try, bmin, bmax)
            b = min((-bmin + b_star) / (bmax - bmin), 1)
            bound = max(func(np.ceil(b * t - 0.5) / t, np.ceil(b * t - 0.5), t),
                        func(np.floor(mu_hat * t + 0.5) / t, np.floor(mu_hat * t + 0.5), t))

            if log_W_star - bound >= np.log(1 / delta):
                while (m_ub - m_lb) > 0.0001:
                    m_try = (m_ub + m_lb) / 2
                    bmax = 1 / m_try
                    bmin = -1 / (1 - m_try)

                    b_star, log_W_star = find_max_log_wealth_constrained(xs[:t] - m_try, bmin, bmax)
                    if log_W_star - bound >= np.log(1 / delta):
                        m_ub = m_try
                        if self.refine:
                            # to have a refinement of the regret
                            b = min((-bmin + b_star) / (bmax - bmin), 1)
                            bound = max(func(np.ceil(b * t - 0.5) / t, np.ceil(b * t - 0.5), t),
                                        func(np.floor(mu_hat * t + 0.5) / t, np.floor(mu_hat * t + 0.5), t))
                    else:
                        m_lb = m_try

            upper_ci[t] = m_ub
            m_ub_old = m_ub

            # Lower CI
            m_ub = min(m_ub_old, mu_hat)
            m_lb = m_lb_old

            # compute regret
            m_try = m_lb
            bmax = 1 / m_try
            bmin = -1 / (1 - m_try)

            b_star, log_W_star = find_max_log_wealth_constrained(xs[:t] - m_try, bmin, bmax)
            b = min((-bmin + b_star) / (bmax - bmin), 1)
            bound = max(func(np.ceil(b * t - 0.5) / t, np.ceil(b * t - 0.5), t),
                        func(np.floor(mu_hat * t + 0.5) / t, np.floor(mu_hat * t + 0.5), t))

            if log_W_star - bound >= np.log(1 / delta):
                while (m_ub - m_lb) > 0.0001:
                    m_try = (m_ub + m_lb) / 2
                    bmax = 1 / m_try
                    bmin = -1 / (1 - m_try)

                    b_star, log_W_star = find_max_log_wealth_constrained(xs[:t] - m_try, bmin, bmax)
                    if log_W_star - bound >= np.log(1 / delta):
                        m_lb = m_try
                        # uncomment next lines to have a refinement of the regret
                        # b = min((-bmin + b_star) / (bmax - bmin), 1)
                        if self.refine:
                            # to have a refinement of the regret
                            b = min((-bmin + b_star) / (bmax - bmin), 1)
                            bound = max(func(np.ceil(b * t - 0.5) / t, np.ceil(b * t - 0.5), t),
                                        func(np.floor(mu_hat * t + 0.5) / t, np.floor(mu_hat * t + 0.5), t))

                    else:
                        m_ub = m_try

            lower_ci[t] = m_lb
            m_lb_old = m_lb

            if t % log_every == 0:
                end = time.time()
                telapsed.append(end - start)
                if verbose:
                    print(t, end=' ')
                start = end

        return lower_ci, upper_ci, telapsed
