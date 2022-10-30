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

def plot_best_k(x_step_params, y_step_params, k_sign):
    # print(tuple(it.product(np.arange(x_step_params[2]), np.arange(y_step_params[2]))))
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
           y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    K = np.zeros((x_step_params[2], y_step_params[2]))
    for i in range(x_step_params[2]):
        for j in range(y_step_params[2]):
            # следующая версия правильна, но если комплексно сопрячь, то результат не зависит от k_sign
            # K[i, j] = _best_k(X[i, j] + 1j * Y[i, j], k_sign)
            K[i, j] = _best_k(X[i, j] + 1j*Y[i, j]*k_sign, k_sign)
    # Z = [i+k for i, k in it.product(np.arange(x_step_params[2]), np.arange(y_step_params[2]))]
    ax = plt.gca()
    ax.pcolor(X, Y, K,
              cmap='inferno', shading='nearest', alpha=0.5)


def plot_sign_f(f, x_step_params, y_step_params):
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
                    y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    Z = np.sign(f(X, Y))
    fig, ax = plt.subplots(1, 1)
    ax.pcolor(X, Y, Z,
                    cmap='inferno', shading='nearest')


if __name__ == '__main__':
    plot_best_k((-10, 10, 200), (-10, 10, 200), 1)
    plot_best_k((-10, 10, 200), (-10, 10, 200), -1)
    plt.show()