import numpy as np


class Sanov:
    @staticmethod
    def f(m, ys, delta, only_last=False):
        # m: (n, K)
        # ys: (K, T)
        # return: (n, T)
        K, T = ys.shape
        if only_last:
            mu_hat = ys.sum(axis=-1) / T  # (K, )
            switch = (T > 8 * np.pi * (K / np.exp(1)) ** 3).astype(float)
            regret = (1 - switch) * (K * np.log(T + 1)) + switch * ((K - 1) * (np.log(2 * (K - 1) / delta) - np.log(1 / delta)))
            return T * np.einsum('k,nk->n', mu_hat, np.log(mu_hat[np.newaxis, ...] / m)) - regret
        else:
            ts = np.arange(1, T + 1)
            mu_hats = ys.cumsum(axis=-1) / ts  # (K, T)
            switches = (ts > 8 * np.pi * (K / np.exp(1)) ** 3).astype(float)
            regret = (1 - switches) * (K * np.log(ts + 1)) + switches * ((K - 1) * (np.log(2 * (K - 1) / delta) - np.log(1 / delta)))
            return ts * np.einsum('kt,nkt->nt', mu_hats, np.log(mu_hats[np.newaxis, ...] / m[..., np.newaxis])) - regret
