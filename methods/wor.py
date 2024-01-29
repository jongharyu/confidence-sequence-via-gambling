import numpy as np

from methods.base import ConfidenceSequence
from utils.special_functions import multibetaln, multinomln


# for fixed number of balls
class MultiWoRCI(ConfidenceSequence):
    def __init__(self, N, betas=(1 / 2, 1 / 2)):
        super().__init__()
        self.N = N  # total number of balls
        self.betas = np.array(betas)

    def f(self, ns, ys, mode='kt', eps=0, only_last=False):
        # ns: (n, M)
        # ys: a sequence of M-dim. vectors; (M, T)
        assert mode in ['kt', 'dirmul']
        if only_last:
            ks = ys.sum(axis=-1)  # (M, )
            diff = ns - ks[np.newaxis, :]  # (n, M)
            if mode == 'kt':
                # based on the KT wealth martingale from horse race
                mask = ((diff < 0).sum(axis=1)).astype(bool).astype(float)  # (n, )
                mask[mask == 1.] = np.inf
                switch = mask - (ks * (np.log(diff) - np.log(self.N - ys.shape[1]))).sum(axis=1)
            else:
                # based on the Dirichlet Multinomial prior-posterior ratio martingale
                switch = (multinomln(ns, axis=1) - multinomln(diff, axis=1))

            return np.nan_to_num(
                switch + (multibetaln(ks + self.betas) - multibetaln(self.betas)),
                nan=1e3, posinf=1e3, neginf=-1e3,
            )  # (n, )
        else:
            csys = ys.cumsum(axis=-1)  # (M, T)

            diff = ns[..., np.newaxis] - csys[np.newaxis, ...]  # (n, M, T)
            if mode == 'kt':
                # based on the KT wealth martingale from horse race
                mask = ((diff < 0).sum(axis=1)).astype(bool).astype(float)  # (n, T)
                mask[mask == 1.] = np.inf
                switch = mask - (csys[np.newaxis, ...] * (np.log(diff) - np.log(self.N - ys.shape[1]))).sum(axis=1)
            else:
                # based on the Dirichlet Multinomial prior-posterior ratio martingale
                switch = (multinomln(ns[..., np.newaxis], axis=1) - multinomln(diff, axis=1))

            return np.nan_to_num(
                switch + (multibetaln(csys + self.betas[:, np.newaxis]) - multibetaln(self.betas)),
                nan=1e3, posinf=1e3, neginf=-1e3,
            )  # (n, T)
