import time

import numpy as np

from methods.base import ConfidenceSequence
from utils import confidence_interval
from precise_co96 import newton_1d_bnd


class PRECiSE_CO96(ConfidenceSequence):
    """
    PRECISE_R70    Portfolio REgret for Confidence SEquences using Robbins [1970].
    [L,U] = PRECISE_R70(X,delta) produces two matrices, of the same
    dimension as X and with lower and upper confidence sequences with
    probability of error delta. X is numer-of-samples by number-of-repetitions.

    This algorithm is described in  Orabona and Jun, "Tight Concentrations
    and Confidence Sequences from the Regret of Universal Portfolio", ArXiv 2021.
    """
    def __init__(self, refine=True):
        super().__init__()
        self.refine = refine

    @confidence_interval
    def construct(delta, xs, eps=1e-4, verbose=False, log_every=100, **kwargs):
        lower_ci = np.zeros_like(xs)
        upper_ci = np.zeros_like(xs)

        mn = np.inf
        mx = -np.inf

        m_lb_old = np.finfo(float).eps
        m_ub_old = 1 - np.finfo(float).eps

        telapsed = []
        start = time.time()
        for t in range(len(xs)):
            mean_c = np.mean(xs[:t + 1])

            # upper confidence interval
            m_ub = m_ub_old
            m_lb = max(m_lb_old, mean_c)

            mn = min(xs[t], mn)
            mx = max(xs[t], mx)

            # calculate regret
            m_try = m_ub
            log_W_star = find_max_log_wealth_constrained_lil(xs[:t + 1] - m_try, mn - m_try, mx - m_try)
            if log_W_star >= np.log(1 / delta):
                while (m_ub - m_lb) > eps:
                    m_try = (m_ub + m_lb) / 2
                    log_W_star = find_max_log_wealth_constrained_lil(xs[:t + 1] - m_try, mn - m_try, mx - m_try)
                    if log_W_star >= np.log(1 / delta):
                        m_ub = m_try
                    else:
                        m_lb = m_try
            upper_ci[t] = m_ub
            m_ub_old = m_ub

            # lower confidence interval
            m_ub = min(m_ub, mean_c)
            m_lb = m_lb_old

            # calculate regret
            m_try = m_lb
            log_W_star = find_max_log_wealth_constrained_lil(xs[:t + 1] - m_try, mn - m_try, mx - m_try)
            if log_W_star >= np.log(1 / delta):
                while (m_ub - m_lb) > eps:
                    m_try = (m_ub + m_lb) / 2
                    log_W_star = find_max_log_wealth_constrained_lil(xs[:t + 1] - m_try, mn - m_try, mx - m_try)
                    if log_W_star >= np.log(1 / delta):
                        m_lb = m_try
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


def find_max_log_wealth_constrained_lil(g, mn, mx):
    myf = lambda bet: np.prod(1 + g * bet)
    df = lambda bet: np.sum(g / (1 + g * bet))
    df2 = lambda bet: -np.sum((g / (1 + g * bet)) ** 2)

    betstar, fval = newton_1d_bnd(myf, df, df2, -1, 1)

    pdf = lambda bet: np.log(np.log(6.6) + 1) / (2 * abs(bet) * (1 + np.log(6.6 / abs(bet))) * (np.log(1 + np.log(6.6 / abs(bet)))) ** 2)

    V = np.sum(g ** 2)

    if betstar > 0:
        s = mn
    else:
        s = mx

    absbetstar = np.abs(betstar)
    if absbetstar != 1:
        delta = (1 + min(s * betstar, 0)) / np.sqrt(V)
    else:
        delta = 0
    delta = absbetstar - max(absbetstar - delta, 0)
    fval = np.log(max((fval - 1) / (np.finfo(float).eps + np.log(fval)) * abs(betstar), fval * np.exp(-1 / 2 * delta ** 2 / (1 + min(s * betstar, 0)) ** 2 * V) * delta) * pdf(absbetstar + np.finfo(float).eps))

    return fval
