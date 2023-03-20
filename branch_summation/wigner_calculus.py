import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from numerical_summation.nb_sum_fn import _sum_fn as Fn


# это алгоритм для F normalized. Она должна быть vectorized.
# Fn = lambda r, phi, gamma: sum_fn(r, phi, gamma, 3)
def Fn_zero_gamma(r, phi, gamma, n_sigma):
    return np.exp(r*np.exp(1j*phi) - r)

def np_wigner(alpha_abs, alpha_arg, beta_abs, beta_arg, Gamma, n_sigma):
    print(alpha_abs)
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
     (-alpha_abs ** 2) * np.abs(Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg, Gamma, n_sigma)) ** 2 + \
     alpha_abs ** 4 / 2 * np.abs(Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg + 2 * Gamma, Gamma, n_sigma)) ** 2
    return 2 / np.pi * np.exp(alpha_abs ** 2 - 2 * (alpha_abs - beta_abs)**2) * \
        (additional_sum + partial_sum)


@nb.vectorize('float64(float64, float64, float64, float64, float64, float64)',
              nopython=True, target_backend='cpu', fastmath=False)
def nb_wigner(alpha_abs, alpha_arg, beta_abs, beta_arg, Gamma, n_sigma):
    m_min = max((np.int_(3), np.int_(np.round(alpha_abs ** 2 - n_sigma * alpha_abs))))
    m_max = np.int_(np.round(alpha_abs**2 + n_sigma * alpha_abs))
    m_arr = np.arange(m_min, m_max)
    log_m_arr = np.log(m_arr)
    half_log_2pi = 1/2*np.log(2*np.pi)

    minus_one_powers = 1 - 2 * np.mod(m_arr, 2)
    coefficients_to_exp = -half_log_2pi - 1 / 2 * log_m_arr - m_arr * (log_m_arr - 1 - 2 * np.log(alpha_abs)) - 1/12 / m_arr
    Fn_sq_arr = np.abs(Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg + (2 * m_arr - 2) * Gamma, Gamma, n_sigma)) ** 2

    partial_sum = 0
    for m_shifted in nb.prange(m_max - m_min):
        partial_sum += minus_one_powers[m_shifted] * np.exp(coefficients_to_exp[m_shifted]) * Fn_sq_arr[m_shifted]
    additional_sum = np.abs(Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg - 2 * Gamma, Gamma, n_sigma)) ** 2 + \
     (-alpha_abs ** 2) * np.abs(Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg, Gamma, n_sigma)) ** 2 + \
     alpha_abs ** 4 / 2 * np.abs(Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg + 2 * Gamma, Gamma, n_sigma)) ** 2
    return 2 / np.pi * np.exp(alpha_abs ** 2 - 2* (alpha_abs - beta_abs)**2) *\
        (additional_sum + partial_sum)


if __name__ == '__main__':
    # print(nb_wigner(10, 1, np.array([10, 2]), np.array([1, 2]), 0, 100))
    alpha_abs = 0.345
    print(np_wigner(alpha_abs, 1, alpha_abs, 1, 0, 10) * np.pi/2)

    x = np.linspace(0.1, 1, 100)
    y = [np_wigner(xi, 1, xi, 1, 0, 10) * np.pi/2 - 1 for xi in x]
    plt.plot(x, y)
    plt.axhline(0, 0, np.max(x))
    plt.show()

    # print(np.log(np.abs(nb_wigner(10, 1, 2, 2.4, 0.01, 8))))
