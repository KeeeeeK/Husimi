import numpy as np
import numpy.typing as npt
import sympy as sp


def re_along_curve(z: sp.Symbol,
                   analytic_func: sp.Expr,
                   points: npt.NDArray[tuple[float, float]]) -> npt.NDArray[float]:
    num_func = sp.lambdify(z, analytic_func)
    return np.array(tuple(map(lambda point: np.real(num_func(point[0] + 1j * point[1])), points)))
