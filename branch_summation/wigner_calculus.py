import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from numerical_summation.nb_sum_fn import _sum_fn as Fn


# это алгоритм для F normalized. Она должна быть vectorized.
# Fn = lambda r, phi, gamma: sum_fn(r, phi, gamma, 3)
def Fn_zero_gamma(r, phi, gamma, n_sigma):
    return np.exp(r * np.exp(1j * phi) - r)


def np_wigner(alpha_abs, alpha_arg, beta_abs, beta_arg, Gamma, n_sigma):
    Fn = Fn_zero_gamma
    type_ = np.double
    m_min = max((np.int_(3), np.int_(np.round(alpha_abs ** 2 - n_sigma * alpha_abs))))
    m_max = np.int_(np.round(alpha_abs ** 2 + n_sigma * alpha_abs))
    m_arr = np.arange(m_min, m_max, dtype=type_)
    log_m_arr = np.log(m_arr)
    half_log_2pi = 1 / 2 * np.log(2 * np.pi)

    minus_one_powers = 1 - 2 * np.mod(m_arr, 2)
    coefficients_to_exp = -half_log_2pi - 1 / 2 * log_m_arr - m_arr * (
            log_m_arr - 1 - 2 * np.log(alpha_abs)) - 1 / 12 / m_arr

    Fn_sq_arr = np.abs(
        Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg + (2 * m_arr - 2) * Gamma, Gamma, n_sigma)) ** 2
    # print(np.exp(coefficients_to_exp - m_arr * 2 * np.log(alpha_abs)) ** -1) # с хорошей точностью это действительно факториалы
    # Fn_sq_arr = type_(Fn_sq_arr)

    partial_sum = 0
    for m_shifted in range(m_max - m_min):
        partial_sum += minus_one_powers[m_shifted] * np.exp(coefficients_to_exp[m_shifted]) * Fn_sq_arr[m_shifted]
    additional_sum = np.abs(Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg - 2 * Gamma, Gamma, n_sigma)) ** 2 + \
                     (-alpha_abs ** 2) * np.abs(
        Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg, Gamma, n_sigma)) ** 2 + \
                     alpha_abs ** 4 / 2 * np.abs(
        Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg + 2 * Gamma, Gamma, n_sigma)) ** 2
    return 2 / np.pi * np.exp(alpha_abs ** 2 - 2 * (alpha_abs - beta_abs) ** 2) * \
        (additional_sum + partial_sum)


@nb.vectorize('float64(float64, float64, float64, float64, float64, float64)',
              nopython=True, target_backend='cpu', fastmath=False)
def nb_wigner(alpha_abs, alpha_arg, beta_abs, beta_arg, Gamma, n_sigma):
    m_min = max((np.int_(3), np.int_(np.round(alpha_abs ** 2 - n_sigma * alpha_abs))))
    m_max = np.int_(np.round(alpha_abs ** 2 + n_sigma * alpha_abs))
    m_arr = np.arange(m_min, m_max)
    log_m_arr = np.log(m_arr)
    half_log_2pi = 1 / 2 * np.log(2 * np.pi)

    minus_one_powers = 1 - 2 * np.mod(m_arr, 2)
    coefficients_to_exp = -half_log_2pi - 1 / 2 * log_m_arr - m_arr * (
            log_m_arr - 1 - 2 * np.log(alpha_abs)) - 1 / 12 / m_arr
    Fn_sq_arr = np.abs(
        Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg + (2 * m_arr - 2) * Gamma, Gamma, n_sigma)) ** 2

    partial_sum = 0
    for m_shifted in nb.prange(m_max - m_min):
        partial_sum += minus_one_powers[m_shifted] * np.exp(coefficients_to_exp[m_shifted]) * Fn_sq_arr[m_shifted]
    additional_sum = np.abs(Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg - 2 * Gamma, Gamma, n_sigma)) ** 2 + \
                     (-alpha_abs ** 2) * np.abs(
        Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg, Gamma, n_sigma)) ** 2 + \
                     alpha_abs ** 4 / 2 * np.abs(
        Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg + 2 * Gamma, Gamma, n_sigma)) ** 2
    return 2 / np.pi * np.exp(alpha_abs ** 2 - 2 * (alpha_abs - beta_abs) ** 2) * \
        (additional_sum + partial_sum)


@nb.vectorize('float64(float64, float64, float64, float64, float64, float64)',
              nopython=True, target_backend='parallel', fastmath=False)
def fast_wigner(alpha_abs, alpha_arg, beta_abs, beta_arg, Gamma, n_sigma):
    Phi0 = alpha_arg - beta_arg - Gamma
    A = 4 * alpha_abs * beta_abs
    norm = 2 * (alpha_abs - beta_abs) ** 2 + A
    sum_fourier = 0

    def summand(k):
        t = np.exp(1j * k * Gamma)
        z = A * t
        AsI_k = np.sqrt(z * z + k * k) - k * np.log(k / z + np.sqrt(1 + (k / z) * (k / z))) - np.log(k * k + z * z) / 4
        return np.exp(1j * k * Phi0 + AsI_k + alpha_abs * alpha_abs * (1 - t * t) - norm)

    if beta_abs < alpha_abs:
        # firstly we should know what to sum
        t_max = np.arccos(beta_abs / alpha_abs)
        k_max = np.int_(t_max / Gamma)
        der2f = -1 / (4 * alpha_abs ** 2) - 4 * Gamma ** 2 * (alpha_abs ** 2 - beta_abs ** 2) + t_max / 8 * \
                (4 * np.sqrt(alpha_abs ** 2 - beta_abs ** 2) + beta_abs * t_max) / (alpha_abs ** 2 * beta_abs)
        k_range = np.int_(1 / np.sqrt(-der2f) * n_sigma)
        # now lets sum
        for k in nb.prange(-k_max - k_range, k_max + k_range + 1):
            sum_fourier += summand(k)
    else:
        # firstly we should know what to sum
        der2f = -1 / (4 * alpha_abs * beta_abs) + 4 * alpha_abs * (alpha_abs - beta_abs) * Gamma ** 2
        k_range = np.int_(1 / np.sqrt(-der2f) * n_sigma)
        # now lets sum
        for k in nb.prange(-k_range, k_range + 1):
            sum_fourier += summand(k)
    return 2 / np.pi / np.sqrt(2 * np.pi) * np.real(sum_K0)


def why_slow_formula_is_bad():
    x = np.linspace(0.1, 1, 100)
    y = [np_wigner(xi, 1, xi, 1, 0, 200) * np.pi / 2 - 1 for xi in x]
    print(np.max(y))
    plt.plot(np.log(x), np.log(np.abs(y)))
    plt.axhline(0, 0, np.max(x))
    plt.show()


if __name__ == '__main__':
    print(fast_wigner(1, 1, 10, 1, 0.01, 5))

    # print(np.log(np.abs(nb_wigner(10, 1, 2, 2.4, 0.01, 8))))
