import matplotlib.pyplot as plt
import numpy as np

from numerical_summation.nb_sum_fn import sum_fn
from branch_summation.wigner_calculus import fast_wigner


def find_husimi(alpha_abs, alpha_arg, beta_abs, beta_arg, Gamma, n_sigma):
    return np.exp(-(alpha_abs - beta_abs) ** 2) / np.pi * \
        np.abs(sum_fn(alpha_abs * beta_abs, 2 * Gamma + alpha_arg - beta_arg, Gamma, n_sigma)) ** 2


def plot_stripe(alpha, gamma, n_sigma=5):
    r_n_dots, phi_n_dots = 100, 100
    n_sigma_r, n_sigma_summation = 3, n_sigma

    # Следующие параметры определяют область пробегания beta
    r_mean = np.abs(alpha)
    r_vals = np.linspace(r_mean - 1.7, r_mean + 1.7, r_n_dots).reshape(-1, 1)
    phi_mean = 2 * r_mean ** 2 * gamma
    phi_range = np.pi / 500
    phi_vals = np.linspace(-phi_range, phi_range, phi_n_dots) + phi_mean
    print('started calcs...')

    # результаты суммирования переводим в точки для графика
    # fn_vals = sum_fn(r_vals * np.abs(alpha), 2 * gamma + np.angle(alpha) - phi_vals, -gamma, n_sigma_summation)
    # res_vals = np.exp(-(np.abs(alpha) - r_vals) ** 2) / np.pi * np.abs(fn_vals) ** 2
    res_vals = fast_wigner(np.abs(alpha), np.angle(alpha), r_vals, phi_vals, gamma, n_sigma_summation)
    print('finished calcs!')
    # координаты точек для графика
    x = r_vals * np.cos(phi_vals - phi_mean)
    y = r_vals * np.sin(phi_vals - phi_mean)
    z = res_vals

    # Сам график
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='inferno')
    plt.show()


if __name__ == '__main__':
    gamma = 10 ** -6
    plot_stripe(1 / np.sqrt(gamma), gamma, n_sigma=5)
