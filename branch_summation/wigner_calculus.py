import numpy as np
import numba as nb

from numerical_summation.nb_sum_fn import _sum_fn as Fn


# это алгоритм для F normalized. Она должна быть vectorized.
# Fn = lambda r, phi, gamma: sum_fn(r, phi, gamma, 3)

def np_wigner(alpha_abs, alpha_arg, beta_abs, beta_arg, Gamma, n_sigma):
    m_min = np.max((np.int_(3), np.int_(np.round(alpha_abs - n_sigma * np.sqrt(alpha_abs)))))
    m_max = np.int_(np.round(alpha_abs + n_sigma * np.sqrt(alpha_abs)))
    m_arr = np.arange(m_min, m_max)
    log_m_arr = np.log(m_arr)

    minus_one_powers = 2 * np.mod(m_arr, 2) - 1
    coefficients_to_exp = - 1 / 2 * log_m_arr - m_arr * (log_m_arr - 1 - 2 * np.log(alpha_abs)) - 1 / m_arr
    Fn_arr = Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg + (2 * m_arr - 2) * Gamma, Gamma, n_sigma)

    partial_sum = np.sum(minus_one_powers * np.exp(coefficients_to_exp) * Fn_arr)
    return 2 / np.pi * np.exp(-beta_abs ** 2 - (alpha_abs - beta_abs) ** 2) * np.abs(partial_sum) ** 2


@nb.vectorize('float64(float64, float64, float64, float64, float64, float64)',
              nopython=True, target_backend='cpu', fastmath=False)
def nb_wigner(alpha_abs, alpha_arg, beta_abs, beta_arg, Gamma, n_sigma):
    m_min = max((np.int_(3), np.int_(np.round(alpha_abs - n_sigma * np.sqrt(alpha_abs)))))
    m_max = np.int_(np.round(alpha_abs + n_sigma * np.sqrt(alpha_abs)))
    m_arr = np.arange(m_min, m_max)
    log_m_arr = np.log(m_arr)

    minus_one_powers = 2 * np.mod(m_arr, 2) - 1
    coefficients_to_exp = - 1 / 2 * log_m_arr - m_arr * (log_m_arr - 1 - 2 * np.log(alpha_abs)) - 1 / m_arr
    Fn_arr = Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg + (2 * m_arr - 2) * Gamma, Gamma, n_sigma)

    partial_sum = 0
    for m_shifted in nb.prange(m_max - m_min):
        partial_sum += minus_one_powers[m_shifted] * np.exp(coefficients_to_exp[m_shifted]) * Fn_arr[m_shifted]
    return 2 / np.pi * np.exp(-beta_abs ** 2 - (alpha_abs - beta_abs) ** 2) * np.abs(partial_sum) ** 2


if __name__ == '__main__':
    print(nb_wigner(10, 1, np.array([10, 2]), np.array([1, 2]), 0.01, 2))
    print(nb_wigner(10, 1, 2, 2.4, 0.01, 2))
