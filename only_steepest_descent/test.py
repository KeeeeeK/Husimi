import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy as sc
from constant_phase_curve import constant_phase_curve
from re_along_curve import analytic_re_along_curve, numeric_re_along_curve
from plot import plot_curve, plot_point, plot_values, plot_scatter
from F_research import F_const_phase_curve, F_decent_point, F, F_values_in_saddle_points


# как это должно быть использовано

def test_constant_phase_curve():
    z = sp.symbols('z')
    analytic_func = sp.atan(z) + z ** 2 - z
    x0, y0 = 0, 0
    step = 0.01
    steps_backward = 1000
    steps_forward = 1000
    start_point = x0, y0
    steps_params = step, steps_backward, steps_forward

    points = constant_phase_curve(z, analytic_func, start_point, steps_params)
    re_values = analytic_re_along_curve(z, analytic_func, points)

    print(re_values)
    plot_curve(points)
    plt.show()


def test_F_consant_phase_curve():
    alpha = 100
    gamma = 1
    # k = 5
    steps_params = (0.1, 200, 200)
    for k in range(-3, 1):
        start_point = F_decent_point(alpha * gamma, k)
        plot_point(start_point)
        points = F_const_phase_curve(alpha * gamma, k, steps_params=steps_params)
        plot_curve(points)
    plt.show()


def test_values_along_curve():
    alpha = 2 * np.exp(1j)
    gamma = 0.01
    k = 1
    steps_params = (0.1, 30, 30)
    points_on_curve = F_const_phase_curve(alpha * gamma, k, steps_params=steps_params)
    values = numeric_re_along_curve(F(alpha * gamma), points_on_curve)
    # print((values[0]))
    plot_values(steps_params, values)
    plt.show()


def test_location_of_saddle_points():
    n_dots = 101
    alpha_gamma = 10
    points = np.array([F_decent_point(alpha_gamma, -k) for k in range(n_dots)])
    plot_scatter(points)
    plt.show()


def test_values_in_saddle_points():
    n_dots = 10
    steps_params = (1, 0, n_dots - 1)
    for alpha_gamma in range(10):
        values = F_values_in_saddle_points(n_dots, alpha_gamma)
        plot_values(steps_params, values, linewidth=alpha_gamma * 0.5)
    plt.show()


def test_maxes_of_saddle_points():
    alpha_gamma_array = np.linspace(10, 10000, 30)
    # k - номер ветви W_k(z)
    k_max_lst = [np.argmax(F_values_in_saddle_points(int(alpha_gamma), alpha_gamma)) for alpha_gamma in
                 alpha_gamma_array]
    plot_scatter(np.array(tuple(zip(alpha_gamma_array, k_max_lst))))
    plt.show()


def test_one_great_saddle_point():
    # Это очень забавно, но походу пик всё время достигается в очень странной точке под номером 0.3183098881*alpha_gamma
    alpha_gamma = 10 ** 10
    min_slope, max_slope = 0.318309880, 0.3183885
    points = np.array([F_decent_point(alpha_gamma, -k)
                       for k in range(int(alpha_gamma * min_slope), int(alpha_gamma * max_slope))])
    k_max = int(alpha_gamma * min_slope) + np.argmax(numeric_re_along_curve(F(alpha_gamma), points))
    print(k_max / alpha_gamma)  # 0.3183098881


def test_z_k_asymptotic():
    pi, ln = np.pi, np.log
    r = 10 ** 7
    phi = 1
    alpha_gamma = 1j / 2 * r * np.exp(1j * phi)
    K = -int(10 ** 6 / 2 / pi)
    # z_K = -pi*K + delta_K
    delta_K_real = 1j / 2 * sc.special.lambertw(-2 * 1j * alpha_gamma, K) + pi * K
    delta_K_asymptotic = 1j / 2 * ln(r / (2 * pi * np.abs(K))) - phi / 2 + np.sign(K) * pi / 4 + \
                         ln(2 * pi * np.abs(K)) / 4 / pi / K - \
                         1j / 4 * phi / pi / K + 1j / 8 / np.abs(K) - ln(r) / 4 / pi / K
    coord = lambda z: (np.real(z), np.imag(z))
    print(coord(delta_K_real),
          coord((delta_K_real - delta_K_asymptotic)),
          ln(r) / r,
          sep='\n')
    z_k = -pi*K + delta_K_real
    print(F_decent_point(alpha_gamma, K)[0] - np.real(z_k*(1-1j*z_k)))


def test_re_f_z_k():
    alpha_gamma_mod = 10 ** 4
    r = alpha_gamma_mod * 2
    k = - int(np.abs(alpha_gamma_mod) / np.pi) * 4
    phi = np.linspace(-np.pi, np.pi, 1000)
    alpha_gamma_array = 1j / 2 * r * np.exp(1j * phi)
    # значение F в самой максимальной перевальной точке
    F_values = np.array([F(alpha_gamma_array[i])(F_decent_point(alpha_gamma_array[i], k)[0] + 1j * F_decent_point(alpha_gamma_array[i], k)[1])
                         for i in range(len(alpha_gamma_array))])

    pi, ln = np.pi, np.log
    big_ln = ln(2 * pi * np.abs(k) / r)
    F_asymptotic = -1j * pi ** 2 * k ** 2 \
                   - pi * k + pi * k * big_ln - 1j * pi * k * (phi - pi / 2 * np.sign(k)) \
                   -1j/4*(phi - pi/2*np.sign(k))**2 + 1/2*big_ln*(phi - pi/2*np.sign(k)+1j/2*big_ln)

    z_k = 1j / 2 * sc.special.lambertw(-2 * 1j * alpha_gamma_array, k)
    F_expected = -1j * z_k ** 2 + z_k
    plot_scatter(np.array(tuple(zip(phi, np.imag(F_asymptotic - F_expected)/pi))), color='blue')
    plot_scatter(np.array(tuple(zip(phi, np.imag(F_expected - F_values)))), color='green')
    plt.show()


if __name__ == '__main__':
    test_re_f_z_k()
