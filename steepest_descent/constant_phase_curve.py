from typing import Tuple, Any

import scipy as sc
import sympy as sp
import numpy as np
from numpy import ndarray


def constant_phase_curve(z: sp.Symbol,
                         analytic_func: sp.Expr,
                         start_point: tuple[int, int],
                         steps_params=tuple[float, int, int]) -> np.ndarray:
    """Даёт координаты точек вдоль кривой постоянной фазы функции exp(analytic_func).

    :param z: аргумент следующей аналитической фукнции
    :param analytic_func: аналитическая функция от z.
    :param start_point: x0, y0.
    Точка перевала от которой будет строиться кривая.
    То есть именно в этой точке func'' зануляется.
    Подразумевается, что кривая постоянной фазы не будет проходить через какие-либо иные перевальные точки.
    :param steps_params: step, steps_backward, steps_forward
    :var step - примерный размер шага вдоль кривой. Требуется сделать его сравнительно маленьким.
    :var steps_backward, steps_forward - количество шагов вперёд и назад вдоль искомой кривой.
    Направлением "вперёд" считается то направление, которое соответствует положительной проекции на ось Re(z)
    (а также Re(f(z)) спадает, Im(f(z)) константа)
    TODO: сделать пояснительную картинку

    :return: Массив из пар координат кривой постоянной фазы.
    Attention!!! Алгоритм подразумевает, что
    """
    x0, y0 = start_point
    step, steps_backward, steps_forward = steps_params

    derivative = sp.diff(analytic_func, )
    cos_chi, sin_chi = _positive_direction(z, derivative, (x0, y0))

    num_func, num_derivative = sp.lambdify(z, analytic_func), sp.lambdify(z, derivative)
    forward_points = _step_algorithm(num_func, num_derivative, (x0, y0),
                                     (+cos_chi, +sin_chi), step, steps_forward)
    backward_points = _step_algorithm(num_func, num_derivative, (x0, y0),
                                      (-cos_chi, -sin_chi), step, steps_backward)
    return np.concatenate((backward_points[::-1], ((x0, y0),), forward_points))


def _positive_direction(z, derivative, start_point, print_derivative=False) -> tuple[float, float]:
    """Положительное направление = направление, вдоль которого:
    1. Соответствует положительной проекции на ось Re(z)
    2. Re(f(z)) спадает
    3. Im(f(z)) константа
    :return cos_chi, sin_chi"""
    x0, y0 = start_point
    z0: complex = x0 + 1j * y0
    if print_derivative is True:
        print(f'derivative in start point = {complex(derivative.evalf(subs={z: z0}))}')
    if not np.isclose(np.complex_(derivative.evalf(subs={z: z0})), 0):
        raise NonZeroFPrime('Алгоритм ожидает, что в перевальной точке нулевая первая производная')
    second_derivative = np.complex_(sp.diff(derivative, z).evalf(subs={z: z0}))
    if np.isclose(second_derivative, 0):
        raise ZeroFPrimePrime('Алгоритм ожидает, что в перевальной точке ненулевая вторая производная')
    cos_2chi = - np.abs(np.real(second_derivative)) / np.abs(second_derivative)
    sin_chi_sign = np.sign(np.imag(second_derivative)) if np.imag(second_derivative) != 0 else 1
    return np.sqrt(1 / 2 * (1 + cos_2chi)), np.sqrt(1 / 2 * (1 - cos_2chi)) * sin_chi_sign


def _step_algorithm(num_func, num_derivative, initial_point, initial_direction, step, num_steps) -> np.ndarray:
    """
    :param num_func: complex -> complex, функция f
    :param num_derivative: complex -> complex, её производная
    :return: точки в формате (x, y) на кривой постоянной фазы функции exp(f(z))
    """
    x0, y0 = initial_point
    phase = np.imag(num_func(x0 + 1j * y0))

    filling_arr = np.empty((num_steps, 2))
    current_point, current_direction = initial_point, initial_direction
    for i in range(num_steps):
        current_point, current_direction = \
            _next_point_and_direction(num_func, num_derivative, current_point, current_direction, step, phase)
        filling_arr[i][0], filling_arr[i][1] = current_point
    return np.array(filling_arr)


def _next_point_and_direction(num_func, num_derivative, current_point, current_direction, step, phase) -> \
        tuple[tuple[float, float], tuple[float, float]]:
    cos_chi, sin_chi = current_direction

    z_exact = _exact_next_z(num_func, num_derivative, current_point, current_direction, step, phase)
    f_prime = num_derivative(z_exact)
    mb_cos_chi, mb_sin_chi = np.real(f_prime) / np.abs(f_prime), -np.imag(f_prime) / np.abs(f_prime)
    if np.abs(mb_cos_chi - cos_chi) + np.abs(mb_sin_chi - sin_chi) > \
            np.abs(mb_cos_chi + cos_chi) + np.abs(mb_sin_chi + sin_chi):
        mb_cos_chi, mb_sin_chi = -mb_cos_chi, -mb_sin_chi

    return (np.real(z_exact), np.imag(z_exact)), (mb_cos_chi, mb_sin_chi)


def _exact_next_z(num_func, num_derivative, current_point, current_direction, step, phase) -> np.complex_:
    x0, y0 = current_point
    cos_chi, sin_chi = current_direction

    # Ожидаемые (estimated) координаты следующей точки
    # (Реальное положение точки на кривой постоянной ширины может отличаться)
    x_es, y_es = x0 + step * cos_chi, y0 + step * sin_chi
    # Натурально параметризованная прямая, проходящая через ожидаемую точку и перпендикулярная исходному направлению
    z_of_t = lambda t: x_es + sin_chi * t + 1j * (y_es - cos_chi * t)
    # Это вариант решения без использования производной
    t_sol = sc.optimize.root_scalar(lambda t: np.imag(num_func(z_of_t(t))) - phase, x0=-step, x1=step)
    # А это вариант с использованием производной <-- в ходе тестов оказался забагованным(((
    # Иногда полученная кривая постоянной ширины искривлялась до неузнаваемости
    # t_sol = sc.optimize.root_scalar(lambda t: np.imag(num_func(z_of_t(t))) - phase, x0=0,
    #                                 fprime=lambda t: np.imag(num_derivative(z_of_t(t) * (sin_chi - 1j * cos_chi))))
    return np.complex_(z_of_t(t_sol.root))


class NonZeroFPrime(Exception):
    pass


class ZeroFPrimePrime(Exception):
    pass
