import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def _next_point_and_direction():
    ...


def _step_algorithm():
    ...


def _solve_curve(f, step):
    ...


def _main_branch_curves(k: int, epsilon: float):
    ...


def plot_main_branches_by_z():
    ...

def plot_sign_f(f, x_step_params, y_step_params):
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
                    y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    Z = np.sign(f(X, Y))
    fig, ax = plt.subplots(1, 1)
    pcm = ax.pcolor(X, Y, Z,
                    cmap='inferno', shading='nearest')


if __name__ == '__main__':
    plot_sign_f(lambda x, y: x**2 + y**2 - 1, (-2, 2, 100), (-3, 3, 200))
    plt.show()
