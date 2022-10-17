import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

def sum_integrals_by_k(Z: complex | float,
                       Gamma: float):
    lambda_var = 1/2/Gamma
    A = 1j * Z / Gamma / 2
    kmax = 5
    k_set = range(0, kmax, np.sign(kmax))
    f_in_z_k_func = lambda z: z_k ** 2 / 2j + z_k
    z_k = np.array([sc.special.lambertw(Z, k=k) for k in k_set])
    f_in_z_k = f_in_z_k_func(z_k)
    print((f_in_z_k * lambda_var - np.abs(A))[[0, 1, 2, 3]])
    # return (np.exp((f_in_z_k - np.abs(A)) * lambda_var))/np.sqrt(Gamma)
    return np.sum(np.exp(f_in_z_k * lambda_var - np.abs(A))) / np.sqrt(2*Gamma)


def simple_graph(R: float, Gamma: float):
    phi_mean, phi_sigma = 0.9, 0.1
    phi_set = np.linspace(phi_mean - phi_sigma, phi_mean + phi_sigma, 10**3)
    y_set = np.array([sum_integrals_by_k(R* np.exp(1j * phi), Gamma) for phi in phi_set])
    plt.plot(phi_set, np.abs(y_set))

if __name__ == '__main__':
    simple_graph(1, 10**-3/4)
    plt.show()