import numpy as np
import matplotlib.pyplot as plt
from only_steepest_descent.plot import plot_beauty
from numerical_summation.nb_sum_fn import sum_fn
from branch_summation.wigner_calculus import nb_wigner as wigner
import numba as nb


@nb.vectorize('complex128(float64, float64, float64, float64)', nopython=True, target_backend='cpu', fastmath=False)
def alternative_sum_fn(r, phi, gamma_kf, gamma_nf):
    gamma_k, gamma_n = np.int_(gamma_kf), np.int_(gamma_nf)
    sum_numer, sum_denom = 0, 0
    gamma = 2 * np.pi * gamma_k / gamma_n
    for j in nb.prange(gamma_n):
        sum_numer += np.exp(-1j * j * j * gamma) * np.exp(r * (np.exp(1j * (phi + 2 * j * gamma)) - 1))
        sum_denom += np.exp(-1j * j * j * gamma)
    return sum_numer / sum_denom


def find_husimi(alpha, beta_conj, gamma, n_sigma):
    return np.exp(-(np.abs(alpha) - np.abs(beta_conj)) ** 2) / np.pi * \
        np.abs(sum_fn(np.abs(alpha) * np.abs(beta_conj), 2 * gamma + np.angle(alpha * beta_conj), -gamma, n_sigma)) ** 2


def alternative_find_husimi(alpha, beta_conj, gamma_k, gamma_n):
    gamma = 2 * np.pi * gamma_k / gamma_n
    return np.exp(-(np.abs(alpha) - np.abs(beta_conj)) ** 2) / np.pi * \
        np.abs(alternative_sum_fn(np.abs(alpha) * np.abs(beta_conj),
                                  2 * gamma + np.angle(alpha * beta_conj), gamma_k, gamma_n)) ** 2


def husimi_values(alpha, gamma, n_sigma, xy_range, freq):
    x_range, y_range = xy_range, xy_range
    x_step_params, y_step_params = (-x_range, x_range, freq), (-y_range, y_range, freq)
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
           y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    Q = find_husimi(alpha, X - 1j * Y, gamma, n_sigma)
    return X, Y, Q


def alternative_husimi_values(alpha, gamma_k, gamma_n, xy_range, freq):
    x_range, y_range = xy_range, xy_range
    x_step_params, y_step_params = (-x_range, x_range, freq), (-y_range, y_range, freq)
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
           y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    Q = alternative_find_husimi(alpha, X - 1j * Y, gamma_k, gamma_n)
    return X, Y, Q


def wigner_values(alpha, gamma, n_sigma, xy_range, freq):
    x_range, y_range = xy_range, xy_range
    x_step_params, y_step_params = (-x_range, x_range, freq), (-y_range, y_range, freq)
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
           y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    W = wigner(np.abs(alpha), np.angle(alpha), np.abs(X + 1j * Y), np.angle(X + 1j * Y), gamma, n_sigma)
    return X, Y, W / 1000


def plot_3d(X, Y, Z):
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='inferno')


def plot_contour(X, Y, Z):
    plt.contour(X, Y, Z, levels=np.array((2.4, 3, 4, 5)) / 100)


if __name__ == '__main__':
    # print(alternative_sum_fn(1, 1, nb.int32(1), nb.int32(12)))
    # plot_contour(*husimi_values(3, np.pi/6, 20, 4.3, 200))
    plot_contour(*alternative_husimi_values(3, 1, 12, 4.3, 500))
    plt.show()
