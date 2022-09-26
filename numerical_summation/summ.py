import numpy as np
import numba as nb

#
# psi = exp(i*gamma)


def _sum_F(r: np.float_, phi: np.float_, gamma: np.float_, n_sigma: np.float_):
    n_min = np.max(2, np.round(r - n_sigma * np.sqrt(r)))
    n_max = np.round(r + n_sigma * np.sqrt(r))
    return np.sum((_ln_summand(n) for n in np.arange(n_min, n_max)))


def _summand(r):
    return 1
