import numpy as np

from numerical_summation.nb_sum_fn import sum_fn
# это алгоритм для F normalized. Она должна быть vectorized.
Fn = lambda r, phi, gamma: sum_fn(r, phi, gamma, 3)

def wigner(alpha_abs, alpha_arg, beta_abs, beta_arg, Gamma, n_sigma):
    m_min = np.max((np.int_(3), np.int_(np.round(alpha_abs - n_sigma * np.sqrt(alpha_abs)))))
    m_max = np.int_(np.round(alpha_abs + n_sigma * np.sqrt(alpha_abs)))
    m_arr = np.arange(m_min, m_max)
    log_m_arr = np.log(m_arr)

    minus_one_powers = 2 * np.mod(m_arr, 2) - 1
    coefficients_to_exp = - 1 / 2 * log_m_arr - m_arr * (log_m_arr - 1 - 2 * np.log(alpha_abs)) - 1 / m_arr
    Fn_arr = Fn(2 * alpha_abs * beta_abs, alpha_arg - beta_arg + (2 * m_arr - 2) * Gamma, Gamma)

    partial_sum = np.sum(minus_one_powers * np.exp(coefficients_to_exp) * Fn_arr)
    return 2 / np.pi * np.exp(-beta_abs ** 2 - (alpha_abs - beta_abs) ** 2) * np.abs(partial_sum)**2

if __name__ == '__main__':
    print(wigner(10, 0, 10, 0, 0.01, 3))