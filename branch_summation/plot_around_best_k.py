import numpy as np
import matplotlib.pyplot as plt
import scipy as sc


def z_k_approx(Z, k):
    L1 = np.log(Z) + 2 * np.pi * k * 1j
    L2 = np.log(L1)
    return 1j * (L1 - L2 + L2 / L1 +
                 L2 * (-2 + L2) / 2 / L1 ** 2 + 0)
    # L2 * (6 - 9 * L2 + 2 * L2 ** 2) / 6/ L1 ** 3 +
    # L2 * (-12 + 36*L2 -22*L2**2 + 3*L2**3) / 12 / L1**4)


def z_k_exact(Z, k):
    return 1j * sc.special.lambertw(Z, k=k)


def f_in_z_k(Z, k, z_k_func=z_k_approx):
    z_k = z_k_func(Z, k)
    return z_k ** 2 / (2 * 1j) + z_k


def k_optimal(R, Gamma_sign):
    return -int(Gamma_sign * np.round(R / (2 * np.pi)))


def F_plot(R, Gamma):
    k = k_optimal(R, np.sign(Gamma))
    x = np.linspace(-np.pi, np.pi, 100)
    Z = R * np.exp(1j * x)
    y_approx = np.exp((np.real(f_in_z_k(Z, k, z_k_func=z_k_approx)) + R * np.sign(k)) / (2 * Gamma))
    y_exact = np.exp((np.real(f_in_z_k(Z, k, z_k_func=z_k_exact)) + R * np.sign(k)) / (2 * Gamma))
    # / np.abs(np.sqrt(1+W_approx(Z, k)))
    plt.plot(x, y_exact, label='exact')
    plt.plot(x, y_approx, label='approx')
    plt.legend()


F_plot(100, 10 ** -2)
plt.show()
