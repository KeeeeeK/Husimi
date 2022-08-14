import matplotlib.pyplot as plt
import numpy as np
import sympy
from constant_phase_curve import constant_phase_curve
from re_along_curve import analytic_re_along_curve, numeric_re_along_curve
from plot import plot_curve, plot_point, plot_values, plot_scatter
from F_research import F_const_phase_curve, F_decent_point, F, F_values_in_saddle_points
# как это должно быть использовано

def test_constant_phase_curve():
    z = sympy.symbols('z')
    analytic_func = sympy.atan(z) + z**2 - z
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
        start_point = F_decent_point(alpha*gamma, k)
        plot_point(start_point)
        points = F_const_phase_curve(alpha*gamma, k, steps_params=steps_params)
        plot_curve(points)
    plt.show()

def test_values_along_curve():
    alpha = 2
    gamma = 0.01
    k = 1
    steps_params = (0.1, 30, 30)
    points_on_curve = F_const_phase_curve(alpha*gamma, k, steps_params=steps_params)
    values = numeric_re_along_curve(F(alpha*gamma), points_on_curve)
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
        plot_values(steps_params, values, linewidth=alpha_gamma*0.5)
    plt.show()


# def test_max_saddle_point():
#     alpha_gamma_array = np.linspace(10, 1000, 30)
#     k_max_lst = [np.argmax(F_values_in_saddle_points(alpha_gamma, alpha_gamma)) for alpha_gamma in alpha_gamma_array]
#     plot_scatter()

if __name__ == '__main__':
    test_values_in_saddle_points()