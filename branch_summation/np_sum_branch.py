import matplotlib.pyplot as plt
import numpy as np
import scipy as sc


def sum_integrals_by_k(Z: complex | float, Gamma: float):
    lambda_var = 1 / 2 / Gamma
    A = 1j * Z / 2 / Gamma
    kmax = -3
    k_set = range(0, kmax, np.sign(kmax))
    f_in_z_k_func = lambda z: z_k ** 2 / 2j + z_k
    z_k = np.array([1j * sc.special.lambertw(Z, k=k) for k in k_set])
    f_in_z_k = f_in_z_k_func(z_k)
    return np.sum(np.exp(f_in_z_k * lambda_var - np.abs(A)))  \
           * np.sqrt(2*np.pi) / (np.sqrt(lambda_var)) *\
           (np.exp(1j*np.pi/4)/2/np.sqrt(np.pi * Gamma))


def _simple_graph(R: float, Gamma: float):
    phi_around_point = True
    if phi_around_point:
        phi_mean, phi_sigma = np.mod(-R-np.pi/2, 2*np.pi), 0.1
        phi_set = np.linspace(phi_mean - phi_sigma, phi_mean + phi_sigma, 10 ** 5)
    else:
        phi_set = np.linspace(0, 2*np.pi, 10 ** 2)
    y_set = np.array([sum_integrals_by_k(R * np.exp(1j * phi), Gamma) for phi in phi_set])
    plt.plot(phi_set, np.abs(y_set))


if __name__ == '__main__':
    _simple_graph(3, 10 ** -2)
    plt.show()