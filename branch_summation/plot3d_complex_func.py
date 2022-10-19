from typing import Callable, Collection

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc


def xyz_from_func(func: Callable[[np.ndarray], np.ndarray], x: Collection, y: Collection,
                  res_apply: Callable[[np.ndarray], np.ndarray] = np.abs) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Функция для отображения res_func комплекcнозначной функции

    :param func: комплекснозначная функция, например, lambda z: z **2 + exp(1j * z)
    :param x: множество значений действительной части z
    :param y: множество значений мнимой части z
    :param res_apply: функция, которую нужно применить к результирующему значению. Обычно это np.abs, но не обязательно.
    :return: три двумерных массива одной формы: sq_x, sq_y, res.
    Для отображения на графике будет применяться пробегание по всем точкам вида (sq_x[i, j], sq_y[i, j], res[i, j])
    """
    sq_x, sq_y = np.repeat([x], len(y), axis=0), np.repeat([y], len(x), axis=0).transpose()
    return sq_x, sq_y, res_apply(func(sq_x + 1j * sq_y))


def plot_complex_func(func: Callable[[np.ndarray], np.ndarray], x: Collection, y: Collection,
                      res_apply: Callable[[np.ndarray], np.ndarray] = np.abs):
    x, y, z = xyz_from_func(func, x, y, res_apply=res_apply)
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='inferno')


def plot_func_in_points(func: Callable[[np.ndarray], np.ndarray], points: np.ndarray,
                        res_apply: Callable[[np.ndarray], np.ndarray] = np.abs):
    x, y = np.real(points), np.imag(points)
    z = res_apply(func(x + 1j * y))
    plt.gca().scatter(x, y, z, s=100)


if __name__ == '__main__':
    complex_func = lambda z: z ** 2 / 2j + z
    plot_complex_func(complex_func, np.linspace(-40, 40, 10), np.linspace(-40, 40, 10),
                      res_apply=np.real)
    plot_func_in_points(complex_func, np.array([1j * sc.special.lambertw(1, k=k) for k in range(0, 10)]),
                        res_apply=np.real)
    plt.show()
