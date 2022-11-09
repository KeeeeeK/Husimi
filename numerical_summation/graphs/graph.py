import matplotlib.pyplot as plt
import numpy as np

from numerical_summation.nb_sum_fn import sum_fn


def plot_stripe(alpha, gamma):
    r_n_dots, phi_n_dots = 500, 5000
    n_sigma_r, n_sigma_summation = 1, 5

    # Следующие параметры определяют область пробегания beta
    r_mean = np.abs(alpha)
    r_vals = np.linspace(r_mean - np.sqrt(r_mean) * n_sigma_r, r_mean + np.sqrt(r_mean) * n_sigma_r, r_n_dots). \
        reshape(-1, 1)
    phi_vals = np.linspace(0, 2 * np.pi, phi_n_dots)

    # результаты суммирования переводим в точки для графика
    fn_vals = sum_fn(r_vals * np.abs(alpha), 2 * gamma + np.angle(alpha) - phi_vals, -gamma, n_sigma_summation)
    res_vals = np.exp(-(np.abs(alpha) - r_vals) ** 2) / np.pi * np.abs(fn_vals) ** 2

    # координаты точек для графика
    x = r_vals * np.cos(phi_vals)
    y = r_vals * np.sin(phi_vals)
    z = res_vals

    # Сам график
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='inferno')
    plt.show()


if __name__ == '__main__':
    plot_stripe(5, 0)
