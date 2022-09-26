import numpy as np

# Сокращение: fn = F(r * exp(i*phi), gamma) * exp(-r)
# Связь с обозначениями из диплома: psi = exp(i*gamma)

two_pi = 2 * np.pi
half_ln_two_pi = 1 / 2 * np.log(2 * np.pi)


# Эта функция реализует менее требовательный интерфейс, чем _sum_fn
def sum_fn(r, phi, gamma, n_sigma):
    """Ожидается, что r>0, n_sigma>0; phi и gamma действительны"""
    return _sum_fn(np.float_(r), np.float_(phi), np.float_(gamma), np.float_(n_sigma))


def _sum_fn(r: np.float_, phi: np.float_, gamma: np.float_, n_sigma: np.float_):
    # Для первых нескольких слагаемых плохо применять формулу Стирлинга, поэтому их мы просуммируем отдельно
    n_min = np.max((np.int_(3), np.int_(np.round(r - n_sigma * np.sqrt(r)))))
    n_max = np.int_(np.round(r + n_sigma * np.sqrt(r)))

    # # Закомментированная версия более понятная, но менее быстрая, потому что требует дополнительного вызова функции
    # return np.sum((np.exp(_ln_summand_fn(n, r, phi, gamma)) for n in np.arange(n_min, n_max, dtype=np.int_))) + \
    #        np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2)

    # # А эта версия очень хороша по скорости, но умирает по памяти.
    # # При r~10**6, len(phi)~10**6 numpy требует 54ГБ оперативки
    # n = np.arange(n_min, n_max)
    # return np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2) + \
    #        np.sum(np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
    #                      1j * np.mod((phi + gamma * n) * (n + 1), two_pi)), axis=1)

    sum_ = 0
    for n in np.arange(n_min, n_max):
        sum_ += np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
                         1j * np.mod((phi + gamma * n) * (n + 1), two_pi))
    return np.exp(-r) * (1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2) + \
           sum_




def _ln_summand_fn(n: np.int_, r: np.float_, phi: np.float_, gamma: np.float_):
    return n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
           1j * np.mod((phi + gamma * n) * (n + 1), two_pi)
