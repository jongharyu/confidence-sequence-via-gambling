import numpy as np
from scipy.optimize import fsolve


def confidence_interval(func):
    def wrapper(self, delta, xs, *args, **kwargs):
        xs = np.atleast_1d(xs.squeeze())
        lower_ci, upper_ci, *_ = func(self, delta, xs, *args, **kwargs)
        lower_ci = np.maximum.accumulate(np.maximum(np.nan_to_num(lower_ci, nan=0), 0))
        upper_ci = np.minimum.accumulate(np.minimum(np.nan_to_num(upper_ci, nan=1), 1))
        if 'wor' in kwargs and kwargs['wor']:
            # for the "sampling without replacement" scenario
            # see Section 6.2 in (Waudby-Smith and Ramdas, 2021)
            # 'do_not_apply_wor' argument is to avoid applying the WOR transformation twice for HybridUP
            if 'do_not_apply_wor' not in kwargs or ('do_not_apply_wor' in kwargs and not kwargs['do_not_apply_wor']):
                N = len(xs)
                ts = np.arange(1, N + 1).reshape(*xs.shape)
                mu_hats = xs.cumsum(axis=0) / ts
                mu_hats_shifted = np.array([0.] + list(mu_hats[:-1])).reshape(*xs.shape)
                lower_ci = (ts - 1) / N * mu_hats_shifted + (1 - (ts - 1) / N) * lower_ci
                upper_ci = (ts - 1) / N * mu_hats_shifted + (1 - (ts - 1) / N) * upper_ci

        return lower_ci, upper_ci, *_

    return wrapper


class ConfidenceSequence:
    def f(self, *args):
        raise NotImplementedError

    def fprime(self, *args):
        raise NotImplementedError

    @staticmethod
    def empirical_mean(xs):
        ts = np.arange(1, len(xs) + 1)
        ss = xs.cumsum()
        mu_hats = ss / ts
        return mu_hats

    # root finding
    def find_root(self, delta, *args,
                  xmin=-np.inf, xmax=np.inf, xinit=-1,
                  maxiter=100, tol=1e-5, verbose=False):
        assert np.all(xmin > -np.inf) and np.all(xmax < np.inf)

        def f(x):
            return self.f(x, *args) - np.log(1 / delta)

        # return the positive root
        if xinit != -1:
            x = xinit
        else:
            x = (xmax - xmin) / 2  # arbitrary initialization
        cnt = 0
        while 1:
            xprev = x
            x = xprev - (f(xprev)) / self.fprime(xprev, *args)  # Newton--Raphson update
            x = np.minimum(xmax, np.maximum(xmin, x))
            if verbose:
                print('xprev, x:', xprev, x)
            cnt += 1
            if (np.abs(x - xprev) < tol).all() or cnt > maxiter:
                break

        if verbose:
            print('cnt, diff:', cnt, np.abs(x - xprev))
        return x

    def find_root_fsolve(self, delta, *args, xinit):
        def f(x):
            return self.f(x, *args) - np.log(1 / delta)

        return fsolve(f, x0=xinit)

    def find_root_bisect(self, delta, *args,
                         xinits,
                         tol=1e-5,
                         maxiter=16, verbose=False):
        def f(x):
            return self.f(x, *args) - np.log(1 / delta)

        # if using scipy.optimize.bisect
        # try:
        #     return bisect(f, *xinits, xtol=tol)
        # except ValueError:
        #     return -1

        # optional args: maxiter=16, verbose=False
        def compare_signs(f1, f2):
            return np.sign(f1) != np.sign(f2)

        xlow, xhi = xinits
        assert xlow < xhi, (xlow, xhi)

        flow, fhi = f(xlow), f(xhi)
        flow = np.inf if np.isnan(flow) else flow
        fhi = np.inf if np.isnan(fhi) else fhi

        if not compare_signs(flow, fhi):
            if flow > 0 and fhi > 0:
                if verbose:
                    print("Wealth not above the threshold!", (xlow, xhi), (flow, fhi))
            if flow < 0 and fhi < 0:
                return np.nan
            if xlow < 1e-5:
                return 0
            elif xhi > 1 - 1e-5:
                return 1

        cnt = 0
        while 1:
            xmid = (xlow + xhi) / 2
            fmid = f(xmid)
            if np.isnan(fmid):
                fmid = np.inf
            if verbose:
                print("(xlow, xhi)", (xlow, xhi), "(flow, fmid, fhi)", (flow, fmid, fhi))

            if compare_signs(flow, fmid):
                xhi, fhi = xmid, fmid
            elif compare_signs(fmid, fhi):
                xlow, flow = xmid, fmid
            else:
                print("xinits", xinits, "(xlow, xhi)", (xlow, xhi), "(flow, fmid, fhi)", (flow, fmid, fhi))
                raise ValueError

            cnt += 1
            if cnt > maxiter or np.abs(xhi - xlow) < tol:
                break

        if verbose:
            print('cnt, diff:', cnt, np.abs(xhi - xlow))

        root = xlow if flow > 0 else xhi
        assert xinits[0] <= root <= xinits[1]
        if verbose:
            # for debugging
            print("\t=>", xinits, (xlow, flow), (xhi, fhi), root)

        return root
