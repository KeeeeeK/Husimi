import numpy as np
import numpy.typing as npt
import sympy as sp
from typing import Callable


def analytic_re_along_curve(z: sp.Symbol,
                            analytic_func: sp.Expr,
                            points: npt.NDArray[tuple[float, float]]) -> npt.NDArray[float]:
    num_func = sp.lambdify(z, analytic_func)
    return numeric_re_along_curve(num_func, points)


def numeric_re_along_curve(num_func: Callable, points: npt.NDArray[tuple[float, float]]) -> npt.NDArray[float]:
    return np.array(tuple(map(lambda point: np.real(num_func(point[0] + 1j * point[1])), points)))
