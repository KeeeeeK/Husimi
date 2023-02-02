import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy as sc
from constant_phase_curve import constant_phase_curve
from re_along_curve import analytic_re_along_curve, numeric_re_along_curve
from plot import plot_curve, plot_point, plot_values, plot_scatter
from F_research import F_const_phase_curve, F_decent_point, F, F_values_in_saddle_points


def constant_phase_curve_2signs(Z: complex, k_range: np.ndarray):
    steps_params = (0.1, 300, 300)
    x_min, x_max = -20, 20
    y_min, y_max = -10, 15

    plt.figure(figsize=((x_max-x_min)/2/2.54, (y_max-y_min)/2/2.54))
    axes = plt.gca()
    _grid_lines(axes)
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)

    z = sp.Symbol('z')
    analytic_func = z ** 2 / 2j + 1j * Z * sp.exp(1j * z)
    for k in k_range:
        z_k = 1j * sc.special.lambertw(Z, k=k)
        x_k, y_k = np.real(z_k), np.imag(z_k)
        plt.scatter([np.real(z_k)], [np.imag(z_k)], color='red', marker='o')
        if k!=0:
            axes.annotate(f'$k={k}$', xy=(x_k, y_k), xytext=(x_k+0.4, y_k+0.2))
        else:
            axes.annotate(f'$k={k}$', xy=(x_k, y_k), xytext=(x_k + 0.6, y_k-0.25))
        for gamma_sign in (-1, 1):
            points = constant_phase_curve(z, analytic_func*gamma_sign, (x_k, y_k), steps_params=steps_params)
            plt.plot(*zip(*points), color=f'C{0 if gamma_sign == 1 else 1}')




def _grid_lines(axes):
    axes.grid(axis='both', which='major', linestyle='--', linewidth=1)
    axes.grid(axis='both', which='minor', linestyle='--', linewidth=0.5)
    axes.minorticks_on()

if __name__ == '__main__':
    constant_phase_curve_2signs(1+1j, np.arange(-5, 6, 1))
    plt.savefig('v')
