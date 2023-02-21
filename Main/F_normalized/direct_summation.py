import numba as nb
import numpy as np

two_pi = 2 * np.pi
half_ln_two_pi = 1 / 2 * np.log(2 * np.pi)

def sum_fn(r, phi, gamma, n_sigma) -> nb.complex128:
    """r>0, n_sigma>0; phi and gamma should be real.
    r, phi, gamma may be arrays. All used arrays should have the same length.
    """
    return nb_sum_fn(np.float_(r), np.float_(phi), np.float_(gamma), np.float_(n_sigma))


@nb.vectorize('complex128(float64, float64, float64, float64)', nopython=True, target_backend='cpu', fastmath=False)
def nb_sum_fn(r: np.float_, phi: np.float_, gamma: np.float_, n_sigma: np.float_):
    n_min = max((np.int_(3), np.int_(np.round(r - n_sigma * np.sqrt(r)))))
    n_max = np.int_(np.round(r + n_sigma * np.sqrt(r)))

    sum_ = 0
    for n in nb.prange(n_min, n_max):
        sum_ += np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
                       1j * np.mod((phi + gamma * n) * (n + 1), two_pi))
    return np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2) + \
           sum_