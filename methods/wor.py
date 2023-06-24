import numpy as np

from methods.base import ConfidenceSequence
from utils import multibetaln, multinomln


# for fixed number of balls
class MultiWoRCI(ConfidenceSequence):
    def __init__(self, N, betas=(1 / 2, 1 / 2)):
        super().__init__()
        self.N = N  # total number of balls
        self.betas = np.array(betas)

    def f(self, ns, ys, mode='kt', eps=0):
        # ns: (n, M)
        # ys: a sequence of M-dim. vectors; (M, T)
        assert mode in ['kt', 'dirmul']
        ns = ns[..., np.newaxis]  # (n, M, 1)
        csys = ys.cumsum(axis=-1)[np.newaxis, ...]  # (1, M, T)

        diff = ns - csys  # (n, M, T)
        if mode == 'kt':
            # based on the KT wealth martingale from horse race
            mask = ((diff < 0).sum(axis=1)).astype(bool).astype(float)  # (n, T)
            mask[mask == 1.] = np.inf
            switch = mask - (csys * (np.log(diff) - np.log(self.N - ys.shape[1]))).sum(axis=1)
        else:
            # based on the Dirichlet Multinomial prior-posterior ratio martingale
            switch = (multinomln(ns, axis=1) - multinomln(diff, axis=1))

        return np.nan_to_num(
            switch + (multibetaln(csys[0] + self.betas[:, np.newaxis]) - multibetaln(self.betas)),
            nan=1e3, posinf=1e3, neginf=-1e3,
        )  # (n, T)
