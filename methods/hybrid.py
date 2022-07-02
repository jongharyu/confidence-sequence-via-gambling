import numpy as np

from methods.baselines import ConfidenceSeqeunce
from methods.lbup import LowerBoundUniversalPortfolioCI
from methods.up import UniversalPortfolioCI
from utils import confidence_interval


class HybridUPCI(ConfidenceSeqeunce):
    def __init__(self, delta, n=1, betas=(1 / 2, 1 / 2)):
        super().__init__(delta)
        self.n = n
        self.betas = betas

    @confidence_interval
    def construct(self, xs, tup, eps=0, tol=1e-5, verbose=False, log_every=100, **kwargs):
        lower_ci = np.zeros_like(xs).astype(float)
        upper_ci = np.zeros_like(xs).astype(float)

        lower_ci[:tup], upper_ci[:tup], telapsed_up, logweights = UniversalPortfolioCI(
            self.delta, betas=self.betas).construct(
            xs[:tup], eps=eps, tol=tol, verbose=verbose, log_every=log_every, do_not_apply_wor=True, **kwargs)
        lower_ci[tup:], upper_ci[tup:], telapsed_lbup = LowerBoundUniversalPortfolioCI(
            self.delta, self.n, tup=tup, logweights=logweights).construct(
            xs, eps=eps, tol=tol, verbose=verbose, log_every=log_every, do_not_apply_wor=True, **kwargs)

        return lower_ci, upper_ci, telapsed_up + telapsed_lbup
