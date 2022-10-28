import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import itertools as it


def _next_point_and_direction():
    ...


def _step_algorithm():
    ...


def _solve_curve(f, step):
    ...


def _main_branch_curves(k: int, epsilon: float):
    ...


def _best_k(Z, k_sign):
    k_mean_abs, k_range = np.int_(np.abs(Z)/(2*np.pi)), 5
    k_arr = np.arange(np.max((0, k_mean_abs - k_range)), k_mean_abs + k_range) * k_sign
    z_k = np.array([1j * sc.special.lambertw(Z, k=k) for k in k_arr])
    f_in_z_k = z_k ** 2 / 2j + z_k
    k_max = k_arr[np.argmax(np.real(f_in_z_k * (-k_sign)))]
    return k_max

def plot_best_k(x_step_params, y_step_params):
    # print(tuple(it.product(np.arange(x_step_params[2]), np.arange(y_step_params[2]))))
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
           y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    # Z = [i+k for i, k in it.product(np.arange(x_step_params[2]), np.arange(y_step_params[2]))]
    fig, ax = plt.subplots(1, 1)
    ax.pcolor(X, Y, Z,
              cmap='inferno', shading='nearest')


def plot_sign_f(f, x_step_params, y_step_params):
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
                    y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    Z = np.sign(f(X, Y))
    fig, ax = plt.subplots(1, 1)
    ax.pcolor(X, Y, Z,
                    cmap='inferno', shading='nearest')


if __name__ == '__main__':
    # plot_sign_f(lambda x, y: x**2 + y**2 - 1, (-2, 2, 100), (-3, 3, 200))
    print(_best_k(10, 1))
    # plot_best_k((-2, 2, 10), (-3, 3, 20))
    # plt.show()