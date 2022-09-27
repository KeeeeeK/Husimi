import numba as nb
import numpy as np

# Это то же самое, что и np_sum_fn, но реализовано с помощью numba и ... чуть более оптимизировано, но менее читаемо.
# Более подробная версия происходящего есть в numpy версии данного файла.
# Основной вклад в прирост скорости даёт параллелизация вычисления суммы

# В данном файле не обошлось без копипаста, но именно так можно сэкономить на времени компиляции и ...
# shape суммы выражения отличается. Фактически, есть 8 функций.
# Каждая называется в соответствии с аргументами, которые являются списками в них.

two_pi = 2 * np.pi
half_ln_two_pi = 1 / 2 * np.log(2 * np.pi)


# Эта функция реализует менее требовательный интерфейс, чем _sum_fn*
def sum_fn(r, phi, gamma, n_sigma) -> nb.complex128:
    """Ожидается, что r>0, n_sigma>0; phi и gamma действительны.
    r, phi, gamma могут быть числами или массивами. Все используемые массивы должны быть одной длины.
    """
    return _sum_fn(np.float_(r), np.float_(phi), np.float_(gamma), np.float_(n_sigma))


@nb.vectorize('complex128(float64, float64, float64, float64)', nopython=True, target_backend='cpu', fastmath=False)
def _sum_fn(r: np.float_, phi: np.float_, gamma: np.float_, n_sigma: np.float_):
    n_min = max((np.int_(3), np.int_(np.round(r - n_sigma * np.sqrt(r)))))
    n_max = np.int_(np.round(r + n_sigma * np.sqrt(r)))

    sum_ = 0
    for n in nb.prange(n_min, n_max):
        sum_ += np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
                       1j * np.mod((phi + gamma * n) * (n + 1), two_pi))
    return np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2) + \
           sum_