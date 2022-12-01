import matplotlib.pyplot as plt
import numpy as np
import scipy as sc


def saddle_points(Z: complex | float, k_arr):
    return np.array([1j * sc.special.lambertw(Z, k=k) for k in k_arr])


def points_to_k_streak(Z: complex | float, k_arr):
    points = saddle_points(Z, k_arr)
    x_points, y_points = np.real(points), np.imag(points)
    for i in range(len(k_arr)):
        if x_points[i] + y_points[i] + 1 > 0:
            print(k_arr[i])


def plot_saddle_points(Z: complex | float, k_arr):
    points = saddle_points(Z, k_arr)
    x_points, y_points = np.real(points), np.imag(points)
    plt.scatter(x_points, y_points)

def plot_re_in_z_k(Z: complex | float, k_arr):
    points = saddle_points(Z, k_arr)
    f_in_z_k = points **2 / 2j + points
    plt.plot(k_arr, (np.real(f_in_z_k)))


if __name__ == '__main__':
    k_sign = 1
    # R = np.linspace(1, 10, 1)
    # Phi = (R + np.pi / 2) * k_sign
    R = 5

    Phi = np.linspace(-np.pi, np.pi, 100)
    Z = R*np.exp(1j*Phi)
    k_arr = np.arange(-1, 2)

    for k in k_arr:
        z_k = 1j*sc.special.lambertw(Z, k=k)
        x = Phi
        y = 1/2*np.real(-1j*(z_k + 1j)**2)
        plt.plot(x, y)
    plt.axhline(0, c='red')

    plt.show()
