import matplotlib.pyplot as plt
import numpy as np
import scipy as sc


def needed_k_arr(Z_abs: float, Gamma_sign: 1 | -1):
    k_mean: np.int_ = - np.round(Z_abs) * np.int_(Gamma_sign)
    k_delta: np.int_ = np.round(3 * np.sqrt(Z_abs))
    if k_delta <= np.abs(k_mean):
        return k_mean + np.arange(-k_delta, k_delta + 1, dtype=np.int_)
    else:
        if Gamma_sign > 0:
            return np.arange(k_mean - k_delta, 1, dtype=np.int_)
        else:
            return np.arange(0, k_mean + k_delta + 1, dtype=np.int_)


def z_k_arr(Z: complex | float, k_arr: np.ndarray):
    return 1j * np.array(tuple(sc.special.lambertw(Z, k=k) for k in k_arr))


def exp_in_z_k(Gamma: float, z_k_arr: np.ndarray):
    return np.exp((z_k_arr ** 2 / 2j + z_k_arr) / (2 * Gamma))


def gamma_n_plus_half(n: int):
    return np.prod(np.arange(2 * n - 1, 0, -2, dtype=np.int_)) / np.power(2, n) * np.sqrt(np.pi)


def _explicit_sum_integrals_by_k(Z: complex | float, Gamma: float):
    lambda_var = 1 / 2 / Gamma
    A = 1j * Z / 2 / Gamma
    kmax = -2
    k_set = range(0, kmax + np.sign(kmax), np.sign(kmax))
    f_in_z_k_func = lambda z: z_k ** 2 / 2j + z_k
    z_k = np.array([1j * sc.special.lambertw(Z, k=k) for k in k_set])
    f_in_z_k = f_in_z_k_func(z_k)
    return np.sum(np.exp(f_in_z_k * lambda_var - np.abs(A))) \
           * np.sqrt(2 * np.pi) / (np.sqrt(lambda_var)) * \
           (np.exp(1j * np.pi / 4) / 2 / np.sqrt(np.pi * Gamma))


def _simple_graph(R: float, Gamma: float):
    phi_around_point = True
    if phi_around_point:
        phi_mean, phi_sigma = np.mod(-R - np.pi / 2, 2 * np.pi), 0.1
        phi_set = np.linspace(phi_mean - phi_sigma, phi_mean + phi_sigma, 10 ** 5)
    else:
        phi_set = np.linspace(0, 2 * np.pi, 10 ** 2)
    y_set = np.array([_explicit_sum_integrals_by_k(R * np.exp(1j * phi), Gamma) for phi in phi_set])
    plt.plot(phi_set, np.abs(y_set))


if __name__ == '__main__':
    # _simple_graph(3, 10 ** -2)
    # plt.show()
    k_arr = needed_k_arr(1, 1)
