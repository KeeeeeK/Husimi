import scipy as sc
import sympy as sp
import numpy as np
import numpy.typing as npt
from typing import Callable

from constant_phase_curve import constant_phase_curve
from re_along_curve import numeric_re_along_curve

def F(alpha_gamma: complex) -> Callable[[complex], complex]:
    return lambda z: -1j * z ** 2 + alpha_gamma * np.exp(2 * 1j * z)


def F_decent_point(alpha_gamma, k: int) -> tuple[float, float]:
    z_k = 1j / 2 * sc.special.lambertw(-2 * 1j * alpha_gamma, k=k)
    return np.real(z_k), np.imag(z_k)


def F_const_phase_curve(alpha_gamma: complex, k: int, steps_params=(0.1, 100, 100)) \
        -> npt.NDArray[tuple[float, float]]:
    z = sp.Symbol('z')
    # Да, эта строчка отличается от той, что представлена в дипломе, однако что f(z), что gamma*f(z)
    # - имеют одну и ту же кривую постоянной фазы
    analytic_func = -1j * z ** 2 + alpha_gamma * sp.exp(2 * 1j * z)
    x_k, y_k = F_decent_point(alpha_gamma, k)
    return constant_phase_curve(z, analytic_func, (x_k, y_k), steps_params=steps_params)

def F_values_in_saddle_points(n_dots: int, alpha_gamma: complex) -> npt.NDArray[float]:
    """
    :param n_dots: число седловых точек
    :param alpha_gamma: alpha*gamma
    :return: Значения в F в седловых точках под номерами (0,-1, -2,..., -n_dots+1)
    """
    points = np.array([F_decent_point(alpha_gamma, -k) for k in range(n_dots)])
    return numeric_re_along_curve(F(alpha_gamma), points)
