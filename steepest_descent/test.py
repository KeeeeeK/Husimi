import sympy
from constant_phase_curve import constant_phase_curve
# как это должно быть использовано

def test():
    z = sympy.symbols('z')
    analytic_func = z ** 2
    x0, y0 = 0, 0
    step = 1
    steps_backward = 0
    steps_forward = 2
    start_point = x0, y0
    steps_params = step, steps_backward, steps_forward
    points = constant_phase_curve(z, analytic_func, start_point, steps_params)

if __name__ == '__main__':
    test()