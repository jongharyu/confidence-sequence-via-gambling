import math
from itertools import permutations

import matplotlib
import numpy as np
from matplotlib import tri as tri, pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import softmax

from utils.special_functions import multibetaln


def cube_to_simplex(ys, axis=-1):
    # ys: shape (T, M) by default
    y0s = 1 - ys.mean(axis=axis, keepdims=True)
    return np.concatenate([y0s, ys / ys.shape[axis]], axis=axis)


def probability_grid(values, n):
    values = set(values)
    # Check if we can extend the probability distribution with zeros
    with_zero = 0. in values
    values.discard(0.)
    if not values:
        raise StopIteration
    values = list(values)
    for p in _probability_grid_rec(values, n, [], 0.):
        if with_zero:
            # Add necessary zeros
            p += (0.,) * (n - len(p))
        if len(p) == n:
            yield from set(permutations(p))  # faster: more_itertools.distinct_permutations(p)


def _probability_grid_rec(values, n, current, current_sum, eps=1e-10):
    if not values or n <= 0:
        if abs(current_sum - 1.) <= eps:
            yield tuple(current)
    else:
        value, *values = values
        inv = 1. / value
        # Skip this value
        yield from _probability_grid_rec(
            values, n, current, current_sum, eps)
        # Add copies of this value
        precision = round(-math.log10(eps))
        adds = int(round((1. - current_sum) / value, precision))
        for i in range(adds):
            current.append(value)
            current_sum += value
            n -= 1
            yield from _probability_grid_rec(
                values, n, current, current_sum, eps)
        # Remove copies of this value
        if adds > 0:
            del current[-adds:]


def get_num_grid_points(m=3, j=1):
    # assert m >= 3
    if m == 2:
        return j + 1
    elif m == 3:
        return np.sum([i ** 1 for i in range(1, j + 1)])
    else:
        return np.sum([get_num_grid_points(m - 1, j) for j in range(1, j + 1)])


def compute_f_batch(
    f,
    ps,
    threshold=None,
    threshold_mode='upper',
):
    fs = np.clip(f(ps), a_min=-np.inf, a_max=np.inf)
    if threshold:
        if threshold_mode == 'upper':
            fs = (fs <= threshold).astype(float)
        else:
            fs = (fs >= threshold).astype(float)
    return fs.sum(), ps[np.where(fs)]


class Intervals(object):
    def __init__(self, intervals):
        self.intervals = intervals

    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        return np.stack([(a < x[:, i]) & (x[: ,i] < b) for (i, (a, b)) in enumerate(self.intervals)], axis=-1).sum(axis=-1)


class Dirichlet(object):
    def __init__(self, alpha):
        self._alpha = np.array(alpha)
        self._coef = - multibetaln(self._alpha)

    def pdf(self, ps):
        '''Returns pdf value for `ps`.'''
        # ps: shape (M, np)
        return np.exp(self._coef + ((self._alpha - 1) * np.log(ps)).sum(axis=-1))  # (np, )


def generate_beta(t, betas, seed=0):
    np.random.seed(seed)
    return np.random.beta(*betas, size=t)


def generate_dirichlet(t, betas, seed=0):
    np.random.seed(seed)
    return np.random.dirichlet(betas, size=t).T


def generate_logit_normal(t, mean, cov, seed=0):
    np.random.seed(seed)
    y = np.random.multivariate_normal(mean, cov, size=t)
    return softmax(np.hstack([np.ones(t)[:, np.newaxis], y]), axis=1).T  # (K, T)


# for triangle
corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
AREA = 0.5 * 1 * 0.75**0.5
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
# For each corner of the triangle, the pair of other corners
pairs = [corners[np.roll(range(3), -i)[1:]] for i in range(3)]


def tri_area(xy, pair):
    # vectorized version
    deltas = np.tile(pair[np.newaxis, ...], (xy.shape[0], 1, 1)) - xy[:, np.newaxis, :]  # (n, 2, 2)
    return .5 * np.abs(np.cross(deltas[:, 0, :], deltas[:, 1, :]))  # (n, )


def xy2bc(xy, tol=1e-4):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    # xy: shape (n, 2)
    coords = np.stack([tri_area(xy, p) for p in pairs], axis=-1) / AREA
    return np.clip(coords, tol, 1.0 - tol)


def draw_contours_tri2d(
    f,
    nlevels=200,
    subdiv=8,
    threshold=None,
    threshold_mode='upper',
    vmax=30,
    cmap='Reds',
    show=True,
    **kwargs
):
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    xy = np.stack([trimesh.x, trimesh.y], axis=-1)
    pvals = np.clip(f(xy2bc(xy)), a_min=-np.inf, a_max=vmax)

    if threshold:
        if threshold_mode == 'upper':
            pvals = (pvals <= threshold).astype(float)
        else:
            pvals = (pvals >= threshold).astype(float)
        cmap = matplotlib.colors.ListedColormap(['red', 'green'])  # color for False and True
        cmap = LinearSegmentedColormap.from_list('', ['white', 'black'])
        nlevels = 2
        kwargs['vmax'] = None
    if show:
        plt.tricontourf(trimesh, pvals, nlevels, cmap=cmap, **kwargs)
        plt.axis('equal')
        plt.axis('off')
        plt.xlim(0, 1)
        plt.ylim(0, 0.75 ** 0.5)
        plt.colorbar()
        plt.show()
    return pvals.sum() / pvals.size


# for 2d grid
def draw_contours_2d(f, nlevels=500, subdiv=8, threshold=None, vmax=30, cmap='Reds', **kwargs):
    x = np.arange(0, 1, (1 / 2) ** subdiv)
    y = np.arange(0, 1, (1 / 2) ** subdiv)
    x_, y_ = np.meshgrid(x, y)
    xy_ = np.stack([x_, y_], axis=-1).reshape(-1, 2)  # (len(x) * len(y), 2)
    z_grid = np.clip(f(xy_).reshape(len(x), len(y), -1)[..., -1], a_min=-np.inf, a_max=vmax)

    if threshold:
        z_grid = (z_grid < threshold).astype(float)
        cmap = matplotlib.colors.ListedColormap(['red', 'green'])  # color for False and True
        cmap = LinearSegmentedColormap.from_list('', ['white', 'black'])
        levels = 2
        kwargs['vmax'] = None

    plt.contourf(x_, y_, z_grid, levels=nlevels, cmap='Reds', **kwargs)
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.colorbar()
    plt.show()
