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
    # x = np.linspace(-20, 20, 100)
    # plt.plot(x, -x - 1)
    # for Z_abs in np.linspace(0.1, 2**10, 20):
    #     Z = Z_abs
    #     print(Z)
    #     plot_saddle_points(Z, np.arange(0, 2))
    #
    # w = np.pi + np.pi * 1j
    # print(w * np.exp(w))
    # plot_saddle_points(w * np.exp(w), [0, 1])
    # plt.show()
    plot_re_in_z_k(2j, np.arange(-2, 1))
    plt.show()
