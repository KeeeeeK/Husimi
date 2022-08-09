import scipy as sc
import numpy as np


def F_decent_point(alpha_gamma, k: int) -> tuple[float, float]:
    z_k = 1j / 2 * sc.special.lambertw(-2 * 1j * alpha_gamma, k=k)
    return np.real(z_k), np.imag(z_k)

