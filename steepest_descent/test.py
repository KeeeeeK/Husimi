import matplotlib.pyplot as plt
import sympy
from constant_phase_curve import constant_phase_curve
from re_along_curve import re_along_curve
from plot import plot_curve
# как это должно быть использовано

def test():
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

if __name__ == '__main__':
    test()