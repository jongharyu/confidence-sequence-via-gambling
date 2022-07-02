import numpy as np


def binary_entropy(p):
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def confidence_interval(func):
    def wrapper(self, xs, *args, **kwargs):
        xs = np.atleast_1d(xs.squeeze())
        lower_ci, upper_ci, *_ = func(self, xs, *args, **kwargs)
        lower_ci = np.maximum.accumulate(np.maximum(np.nan_to_num(lower_ci, nan=0), 0))
        upper_ci = np.minimum.accumulate(np.minimum(np.nan_to_num(upper_ci, nan=1), 1))
        if 'wor' in kwargs and kwargs['wor']:
            # 'do_not_apply_wor' argument is to avoid applying the WOR transformation twice for HybridUP.
            if 'do_not_apply_wor' not in kwargs or ('do_not_apply_wor' in kwargs and not kwargs['do_not_apply_wor']):
                N = len(xs)
                ts = np.arange(1, N + 1).reshape(*xs.shape)
                mu_hats = xs.cumsum(axis=0) / ts
                mu_hats_shifted = np.array([0.] + list(mu_hats[:-1])).reshape(*xs.shape)
                lower_ci = (ts - 1) / N * mu_hats_shifted + (1 - (ts - 1) / N) * lower_ci
                upper_ci = (ts - 1) / N * mu_hats_shifted + (1 - (ts - 1) / N) * upper_ci

        return lower_ci, upper_ci, *_

    return wrapper
