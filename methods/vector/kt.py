import numpy as np
from scipy.special import logsumexp, betaln

from utils.special_functions import multibetaln


# Two horse races combined via average of wealths for multidim case
class CombinedTwoHorseRacesCS:
    def __init__(self, M=2, betas=(1 / 2, 1 / 2)):
        self.M = M
        self.betas = np.array(betas)

    def f(self, qs, ys, eps=0, only_last=False):
        # qs: (n, M)
        # ys: a sequence of M-dim. vectors; (M, T)
        assert qs.shape[-1] == ys.shape[0], (qs.shape, ys.shape)
        assert ys.shape[0] == self.M

        ts = np.arange(1, ys.shape[1] + 1)  # (T, )
        if only_last:
            ks = ys.sum(axis=-1).reshape(-1, 1)  # (M, 1)
            log_wealth = (
                logsumexp(
                    np.stack(
                        [self.fbase(qs[..., j], ts[-1:], ks[j]) for j in range(self.M)],
                        axis=0,
                    ),
                    axis=0,
                )
                - np.log(self.M)
            ).reshape(-1)  # (n, )
            return log_wealth
        else:
            csys = ys.cumsum(axis=-1)  # (M, T)
            log_wealth = logsumexp(
                np.stack(
                    [self.fbase(qs[..., j], ts, csys[j]) for j in range(self.M)], axis=0
                ),
                axis=0,
            ) - np.log(self.M)  # (n, T)
            return log_wealth

    def fbase(self, m, t, s, eps=0):
        # m: (n, )
        # t: (T, )
        # s: (T, )
        m = m.reshape(-1, 1)  # (n, 1)
        t = t.reshape(1, -1)  # (1, T)
        s = s.reshape(1, -1)  # (1, T)
        # negative log bernoulli probability with count s at time step t
        return (
            -s * np.log(m + eps)
            - (t - s) * np.log(1 - m + eps)
            + betaln(s + self.betas[0], t - s + self.betas[1])
            - betaln(*self.betas)
        )  # (n, T)


class MultiHorseRaceCI:
    def __init__(self, M=2, betas=None):
        self.M = M
        self.betas = 0.5 * np.ones((self.M,)) if betas is None else np.array(betas)

    def f(self, qs, ys, eps=0, only_last=False):
        # qs: (n, M)
        # ys: a sequence of M-dim. vectors; (M, T)
        assert qs.shape[-1] == ys.shape[0]
        assert ys.shape[0] == self.M
        mask = (
            ((qs < 0).sum(axis=-1) + (qs > 1).sum(axis=-1)).astype(bool).astype(float)
        )  # (n, )
        mask[mask == 1.0] = np.inf

        if only_last:
            ks = ys.sum(axis=-1)  # (M, )
            return np.nan_to_num(
                mask
                + multibetaln(ks + self.betas)
                + -multibetaln(self.betas)
                + -np.einsum("m,nm->n", ks, np.log(qs)),
                nan=1e3,
                posinf=1e3,
                neginf=-1e3,
            )  # (n, )
        else:
            csys = ys.cumsum(axis=-1)  # (M, T)
            return np.nan_to_num(
                mask[:, np.newaxis]  # (n, 1)
                + multibetaln(csys + self.betas[:, np.newaxis])
                + -multibetaln(self.betas)
                + -np.einsum("mt,nm->nt", csys, np.log(qs)),
                nan=1e3,
                posinf=1e3,
                neginf=-1e3,
            )  # (n, T)
