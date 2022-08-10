import matplotlib.pyplot as plt
import sympy
from constant_phase_curve import constant_phase_curve
from re_along_curve import re_along_curve
from plot import plot_curve, plot_point
from F_research import F_const_phase_curve, F_decent_point
# как это должно быть использовано

def test_1():
    z = sympy.symbols('z')
    analytic_func = sympy.atan(z) + z**2 - z
    x0, y0 = 0, 0
    step = 0.01
    steps_backward = 1000
    steps_forward = 1000
    start_point = x0, y0
    steps_params = step, steps_backward, steps_forward

    points = constant_phase_curve(z, analytic_func, start_point, steps_params)
    re_values = re_along_curve(z, analytic_func, points)

    print(re_values)
    plot_curve(points)
    plt.show()

def test_2():
    alpha = 10
    gamma = 0.01
    # k = 5
    steps_params = (0.1, 200, 200)
    for k in range(-5, 0):
        start_point = F_decent_point(alpha*gamma, k)
        plot_point(start_point)
        points = F_const_phase_curve(alpha*gamma, k, steps_params=steps_params)
        plot_curve(points)
    plt.show()

if __name__ == '__main__':
    test_2()