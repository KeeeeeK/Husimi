import scipy as sc
import sympy as sp
import numpy as np
import numpy.typing as npt

from constant_phase_curve import constant_phase_curve


def F_decent_point(alpha_gamma, k: int) -> tuple[float, float]:
    z_k = 1j / 2 * sc.special.lambertw(-2 * 1j * alpha_gamma, k=k)
    return np.real(z_k), np.imag(z_k)


def F_const_phase_curve(alpha: complex, gamma: float, k: int, steps_params=(0.1, 100, 100)) \
        -> npt.NDArray[tuple[float, float]]:
    z = sp.Symbol('z')
    # Да, эта строчка отличается от той, что представлена в дипломе, однако что f(z), что gamma*f(z) 
    # - имеют одну и ту же кривую постоянной фазы
    analytic_func = -1j * z ** 2 + alpha * gamma * sp.exp(2 * 1j * z)
    x_k, y_k = F_decent_point(alpha * gamma, k)
    return constant_phase_curve(z, analytic_func, (x_k, y_k), steps_params=steps_params)

