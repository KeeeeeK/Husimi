import numba as nb
import numpy as np

# Это то же самое, что и np_sum_fn, но реализовано с помощью numba и ... чуть более оптимизировано, но менее читаемо.
# Более подробная версия происходящего есть в numpy версии данного файла.
# Основной вклад в прирост скорости даёт параллелизация вычисления суммы

# В данном файле не обошлось без копипаста, но именно так можно сэкономить на времени компиляции и ...
# тип результирующего выражения отличается. Фактически, есть 8 функций. Каждая относится к тому, как реализована сумма

two_pi = 2 * np.pi
half_ln_two_pi = 1 / 2 * np.log(2 * np.pi)


# Эта функция реализует менее требовательный интерфейс, чем _sum_fn_*
def sum_fn(r, phi, gamma, n_sigma):
    """Ожидается, что r>0, n_sigma>0; phi и gamma действительны.
    r, phi, gamma могут быть числами или массивами
    """
    r, phi, gamma, n_sigma = np.float_(r), np.float_(phi), np.float_(gamma), np.float_(n_sigma)
    args = r, phi, gamma, n_sigma
    match r.shape, phi.shape, gamma.shape:
        case (), (), ():
            return _sum_fn(*args)
        case (int(_), ), (), ():
            return _sum_fn_r(*args)
        case (), (int(_), ), ():
            return _sum_fn_phi(*args)
        case (), (), (int(_), ):
            return _sum_fn_gamma(*args)
        case (int(r_len), ), (int(phi_len), ), () if r_len == phi_len:
            return _sum_fn_r_phi(*args)
        case (int(r_len), ), (), (int(gamma_len)) if r_len == gamma_len:
            return _sum_fn_r_gamma(*args)
        case (), (int(phi_len), ), (int(gamma_len)) if phi_len == gamma_len:
            return _sum_fn_phi_gamma(*args)
        case (int(r_len), ), (int(phi_len), ), (int(gamma_len), ) if r_len == phi_len == gamma_len:
            return _sum_fn_r_phi_gamma(*args)
        case _:
            raise Exception('Неожиданный тип аргументов')


@nb.jit('complex128(float64, float64, float64, float64)', parallel=True, nopython=True, cache=True)
def _sum_fn(r: np.float_, phi: np.float_, gamma: np.float_, n_sigma: np.float_):
    n_min = max((np.int_(3), np.int_(np.round(r - n_sigma * np.sqrt(r)))))
    n_max = np.int_(np.round(r + n_sigma * np.sqrt(r)))

    sum_ = 0
    for n in nb.prange(n_min, n_max):
        sum_ += np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
                       1j * np.mod((phi + gamma * n) * (n + 1), two_pi))
    return np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2) + \
           sum_


@nb.jit('complex128[:](float64[:], float64, float64, float64)', parallel=True, nopython=True, cache=True)
def _sum_fn_r(r: np.float_, phi: np.float_, gamma: np.float_, n_sigma: np.float_):
    n_min = max((np.int_(3), np.int_(np.round(r - n_sigma * np.sqrt(r)))))
    n_max = np.int_(np.round(r + n_sigma * np.sqrt(r)))

    sum_ = np.zeros(len(r), dtype=np.complex_)
    for n in nb.prange(n_min, n_max):
        sum_ += np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
                       1j * np.mod((phi + gamma * n) * (n + 1), two_pi))
    return np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2) + \
           sum_


@nb.jit('complex128[:](float64, float64[:], float64, float64)', parallel=True, nopython=True, cache=True)
def _sum_fn_phi(r: np.float_, phi: np.float_, gamma: np.float_, n_sigma: np.float_):
    n_min = max((np.int_(3), np.int_(np.round(r - n_sigma * np.sqrt(r)))))
    n_max = np.int_(np.round(r + n_sigma * np.sqrt(r)))

    sum_ = np.zeros(len(phi), dtype=np.complex_)
    for n in nb.prange(n_min, n_max):
        sum_ += np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
                       1j * np.mod((phi + gamma * n) * (n + 1), two_pi))
    return np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2) + \
           sum_


@nb.jit('complex128[:](float64, float64, float64[:], float64)', parallel=True, nopython=True, cache=True)
def _sum_fn_gamma(r: np.float_, phi: np.float_, gamma: np.float_, n_sigma: np.float_):
    n_min = max((np.int_(3), np.int_(np.round(r - n_sigma * np.sqrt(r)))))
    n_max = np.int_(np.round(r + n_sigma * np.sqrt(r)))

    sum_ = np.zeros(len(gamma), dtype=np.complex_)
    for n in nb.prange(n_min, n_max):
        sum_ += np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
                       1j * np.mod((phi + gamma * n) * (n + 1), two_pi))
    return np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2) + \
           sum_


@nb.jit('complex128[:](float64[:], float64[:], float64, float64)', parallel=True, nopython=True, cache=True)
def _sum_fn_r_phi(r: np.float_, phi: np.float_, gamma: np.float_, n_sigma: np.float_):
    n_min = max((np.int_(3), np.int_(np.round(r - n_sigma * np.sqrt(r)))))
    n_max = np.int_(np.round(r + n_sigma * np.sqrt(r)))

    sum_ = np.zeros(len(phi), dtype=np.complex_)
    for n in nb.prange(n_min, n_max):
        sum_ += np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
                       1j * np.mod((phi + gamma * n) * (n + 1), two_pi))
    return np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2) + \
           sum_


@nb.jit('complex128[:](float64[:], float64, float64[:], float64)', parallel=True, nopython=True, cache=True)
def _sum_fn_r_gamma(r: np.float_, phi: np.float_, gamma: np.float_, n_sigma: np.float_):
    n_min = max((np.int_(3), np.int_(np.round(r - n_sigma * np.sqrt(r)))))
    n_max = np.int_(np.round(r + n_sigma * np.sqrt(r)))

    sum_ = np.zeros(len(r), dtype=np.complex_)
    for n in nb.prange(n_min, n_max):
        sum_ += np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
                       1j * np.mod((phi + gamma * n) * (n + 1), two_pi))
    return np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2) + \
           sum_


@nb.jit('complex128[:](float64, float64[:], float64[:], float64)', parallel=True, nopython=True, cache=True)
def _sum_fn_phi_gamma(r: np.float_, phi: np.float_, gamma: np.float_, n_sigma: np.float_):
    n_min = max((np.int_(3), np.int_(np.round(r - n_sigma * np.sqrt(r)))))
    n_max = np.int_(np.round(r + n_sigma * np.sqrt(r)))

    sum_ = np.zeros(len(phi), dtype=np.complex_)
    for n in nb.prange(n_min, n_max):
        sum_ += np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
                       1j * np.mod((phi + gamma * n) * (n + 1), two_pi))
    return np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2) + \
           sum_


@nb.jit('complex128[:](float64[:], float64[:], float64[:], float64)', parallel=True, nopython=True, cache=True)
def _sum_fn_r_phi_gamma(r: np.float_, phi: np.float_, gamma: np.float_, n_sigma: np.float_):
    n_min = max((np.int_(3), np.int_(np.round(r - n_sigma * np.sqrt(r)))))
    n_max = np.int_(np.round(r + n_sigma * np.sqrt(r)))

    sum_ = np.zeros(len(phi), dtype=np.complex_)
    for n in nb.prange(n_min, n_max):
        sum_ += np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
                       1j * np.mod((phi + gamma * n) * (n + 1), two_pi))
    return np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2) + \
           sum_
