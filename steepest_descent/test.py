import matplotlib.pyplot as plt
import sympy
from constant_phase_curve import constant_phase_curve
from re_along_curve import analytic_re_along_curve, numeric_re_along_curve
from plot import plot_curve, plot_point, plot_values
from F_research import F_const_phase_curve, F_decent_point, F
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
    re_values = analytic_re_along_curve(z, analytic_func, points)

    print(re_values)
    plot_curve(points)
    plt.show()

def test_2():
    alpha = 100
    gamma = 1
    # k = 5
    steps_params = (0.1, 200, 200)
    for k in range(-2, 1):
        start_point = F_decent_point(alpha*gamma, k)
        plot_point(start_point)
        points = F_const_phase_curve(alpha*gamma, k, steps_params=steps_params)
        plot_curve(points)
    plt.show()

def test_3():
    alpha = 2
    gamma = 0.01
    k = -1
    steps_params = (0.1, 30, 30)
    points_on_curve = F_const_phase_curve(alpha*gamma, k, steps_params=steps_params)
    values = numeric_re_along_curve(F(alpha*gamma), points_on_curve)
    # print((values[0]))
    plot_values(steps_params, values)
    plt.show()

if __name__ == '__main__':
    test_3()